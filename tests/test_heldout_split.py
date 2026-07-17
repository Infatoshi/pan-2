"""Held-out stem split (per-checkpoint held-out eval) contracts."""

import zlib

import pytest
import torch

from pan2.config import Config, ModelConfig, TrainConfig
from pan2.data.gpu_pipeline import _is_heldout, apply_split


def _h(stem: str) -> float:
    return (zlib.crc32(stem.encode()) % 10_000) / 10_000


def _items(n: int) -> list[dict]:
    return [{"stem": f"vid{i:05d}"} for i in range(n)]


def test_is_heldout_deterministic():
    first = {s: _is_heldout(s, 0.02) for s in ("abc", "vid00001", "xyz-9")}
    for s, v in first.items():
        assert _is_heldout(s, 0.02) is v
    # boundary: frac 0 puts nothing in val, frac ~1 puts everything
    assert not any(_is_heldout(f"vid{i}", 0.0) for i in range(200))
    assert all(_is_heldout(f"vid{i}", 0.9999) for i in range(200))


def test_split_fraction_sanity():
    stems = [f"stem{i:06d}" for i in range(5000)]
    frac = sum(_is_heldout(s, 0.02) for s in stems) / len(stems)
    assert 0.005 < frac < 0.045  # crc32 spread, no clustering on sequential ids


def test_apply_split_disjoint_and_complete():
    items = _items(2000)
    train = apply_split(items, 0.05, "train")
    val = apply_split(items, 0.05, "val")
    t_stems = {it["stem"] for it in train}
    v_stems = {it["stem"] for it in val}
    assert not (t_stems & v_stems)
    assert t_stems | v_stems == {it["stem"] for it in items}
    assert len(val) > 0


def test_apply_split_noop_when_disabled():
    items = _items(10)
    assert apply_split(items, 0.0, "train") is items


def test_apply_split_rejects_bad_args():
    items = _items(10)
    for frac, split in ((0.0, "val"), (1.0, "train"), (-0.1, "train")):
        with pytest.raises(ValueError):
            apply_split(items, frac, split)
    with pytest.raises(ValueError):
        apply_split(items, 0.5, "holdout")


def test_apply_split_raises_when_val_empty():
    # every stem hashed above this frac -> val set would be empty
    items = [{"stem": "a"}, {"stem": "b"}]
    frac = min(_h("a"), _h("b"))
    with pytest.raises(FileNotFoundError):
        apply_split(items, frac, "val")


def _tiny_state(device: str):
    from pan2.kernels.fused_adamw import build_adamw
    from pan2.models.policy import PanPolicy
    from pan2.train.loop import TrainState

    cfg = Config(
        model=ModelConfig(
            image_size=64,
            d_model=64,
            n_layers=2,
            n_heads=4,
            context_len=8,
            action_chunk=2,
            n_discrete=23,
            frame_subsample=2,
            stem_channels=16,
        ),
        train=TrainConfig(
            stage="pretrain",
            batch_size=4,
            device=device,
            bf16=False,
            n_hard_negatives=2,
        ),
    )
    model = PanPolicy(cfg.model).to(device)
    optim = build_adamw(
        model.parameters(), lr=1e-4, weight_decay=0.0, device_type=device
    )
    return TrainState(model=model, optim=optim, device=torch.device(device)), cfg


def _rand_batch(cfg: Config, device: str):
    b, t, s = cfg.train.batch_size, cfg.model.context_len, cfg.model.image_size
    k = cfg.train.n_hard_negatives
    return {
        "frames": torch.rand(b, t, 3, s, s, device=device),
        "goal": torch.rand(b, 3, s, s, device=device),
        "neg": torch.rand(b, k, 3, s, s, device=device),
    }


def test_eval_contrastive_returns_finite_metrics():
    from pan2.train.loop import eval_contrastive

    state, cfg = _tiny_state("cuda" if torch.cuda.is_available() else "cpu")
    state.model.train()

    def gen():
        while True:
            yield _rand_batch(cfg, state.device.type)

    ev = eval_contrastive(state, cfg, gen(), n_batches=3)
    assert 0.0 < ev["val_loss"] < float("inf")
    assert 0.0 <= ev["val_acc"] <= 1.0
    # chance is 1/(B+K) = 1/6 here; a random net should sit in that ballpark
    assert ev["val_acc"] <= 0.5
    # eval must not leave the model in eval mode or touch the step counter
    assert state.model.training
    assert state.step == 0
    # and no grads were produced
    assert all(p.grad is None for p in state.model.parameters())
