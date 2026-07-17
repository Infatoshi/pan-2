"""Wrong-horizon multi-negative sampling + head/policy shape contracts."""

import numpy as np
import torch

from pan2.config import ModelConfig
from pan2.data.gpu_pipeline import PipelineConfig, _Producer, _subsample_indices
from pan2.models.heads import GoalValueHead
from pan2.models.policy import PanPolicy
from pan2.train.losses import contrastive_loss


def _producer(cfg: PipelineConfig, t_sub: int) -> _Producer:
    class _Ring:
        pass

    ring = _Ring()
    ring.t_sub = t_sub
    p = _Producer.__new__(_Producer)
    p.ring = ring
    p.cfg = cfg
    import random

    p.rng = random.Random(0)
    return p


def test_neg_horizons_single_is_legacy_beyond_window():
    cfg = PipelineConfig(min_goal_horizon=10, max_goal_horizon=150, n_hard_negatives=1)
    p = _producer(cfg, t_sub=1)
    for _ in range(200):
        (h,) = p._sample_neg_horizons(goal_horizon=80)
        assert cfg.max_goal_horizon < h <= 2 * cfg.max_goal_horizon


def test_neg_horizons_wrong_horizon_properties():
    cfg = PipelineConfig(min_goal_horizon=10, max_goal_horizon=150, n_hard_negatives=4)
    p = _producer(cfg, t_sub=1)
    saw_early = saw_late = False
    for _ in range(100):
        goal_h = p.rng.randint(cfg.min_goal_horizon, cfg.max_goal_horizon)
        hs = p._sample_neg_horizons(goal_h)
        assert len(hs) == 4
        for h in hs:
            assert 1 <= h <= 2 * cfg.max_goal_horizon  # inside loaded window
            assert abs(h - goal_h) >= cfg.min_goal_horizon  # wrong horizon
        saw_early |= any(h < goal_h for h in hs)
        saw_late |= any(h > goal_h for h in hs)
    assert saw_early and saw_late  # both too-early and too-late futures occur


def test_make_clip_staging_layout_multi_neg(tmp_path):
    t_full, k_neg, img = 16, 3, 8
    cfg = PipelineConfig(
        context_len=t_full,
        frame_subsample=1,
        image_size=img,
        min_goal_horizon=2,
        max_goal_horizon=4,
        n_hard_negatives=k_neg,
    )
    idxs = _subsample_indices(t_full, 1)
    t_sub = len(idxs) + 1 + k_neg
    p = _producer(cfg, t_sub=t_sub)
    p._staging = torch.empty(t_sub, 3, img, img, dtype=torch.uint8)
    # frame value == frame index, so placement is checkable
    t_total = t_full + 2 * cfg.max_goal_horizon + 4
    frames = np.arange(t_total, dtype=np.uint8)[:, None, None, None]
    frames = np.broadcast_to(frames, (t_total, img, img, 3)).copy()
    npy = tmp_path / "img.npy"
    np.save(npy, frames)
    item = {"source": "npy", "img": str(npy)}
    out = p._make_clip(item, idxs)
    assert out.shape == (t_sub, 3, img, img)
    start = int(out[0, 0, 0, 0])
    # context is frames start..start+t_full-1 at stride 1
    ctx_vals = out[:t_full, 0, 0, 0].tolist()
    assert ctx_vals == list(range(start, start + t_full))
    goal_val = int(out[t_full, 0, 0, 0])
    goal_h = goal_val - (start + t_full - 1)
    assert cfg.min_goal_horizon <= goal_h <= cfg.max_goal_horizon
    for j in range(k_neg):
        nv = int(out[t_full + 1 + j, 0, 0, 0])
        h = nv - (start + t_full - 1)
        assert 1 <= h <= 2 * cfg.max_goal_horizon
        assert abs(h - goal_h) >= 1


def test_value_head_multi_neg_columns():
    torch.manual_seed(0)
    head = GoalValueHead(16)
    b, k = 5, 3
    s = torch.randn(b, 16)
    g = torch.randn(b, 16)
    n = torch.randn(b, k, 16)
    out = head.logits(s, g, n)
    assert out.shape == (b, b + k)
    # [B,D] single neg must match [B,1,D]
    single = head.logits(s, g, n[:, 0])
    single_3d = head.logits(s, g, n[:, :1])
    assert torch.equal(single, single_3d)
    # each extra column is that row's own negative similarity
    sn = head.encode_state(s)
    nn = head.encode_goal(n)
    expect = torch.einsum("bd,bkd->bk", sn, nn) / 0.07
    assert torch.allclose(out[:, b:], expect, atol=1e-6)


def test_policy_forward_multi_neg():
    cfg = ModelConfig(
        image_size=64, d_model=64, n_layers=2, n_heads=4,
        context_len=8, action_chunk=2, n_discrete=23,
        frame_subsample=2, stem_channels=16,
    )
    m = PanPolicy(cfg)
    b, k = 3, 4
    frames = torch.rand(b, cfg.context_len, 3, 64, 64)
    goal = torch.rand(b, 3, 64, 64)
    neg = torch.rand(b, k, 3, 64, 64)
    out = m(frames, goal, neg)
    assert out["contrastive_logits"].shape == (b, b + k)
    loss = contrastive_loss(out["contrastive_logits"])
    loss.backward()
    assert torch.isfinite(loss)
