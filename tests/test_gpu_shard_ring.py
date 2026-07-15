"""Shard-source integration test for the GPU clip ring (skipped without CUDA)."""

import numpy as np
import pytest
import torch

from pan2.data.gpu_pipeline import PipelineConfig, PipelinedGpuPretrainLoader
from pan2.data.shards import ShardWriter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _ramp(t=64, h=8):
    img = np.zeros((t, h, h, 3), dtype=np.uint8)
    for i in range(t):
        img[i] = i
    act = np.zeros((t, 25), dtype=np.float32)
    return img, act


def test_ring_reads_shard_source(tmp_path):
    w = ShardWriter(tmp_path, image_size=8)
    for stem in ("a", "b"):
        img, act = _ramp()
        w.add_episode(img, act, stem)
    w.close()

    cfg = PipelineConfig(
        shards_dir=str(tmp_path),
        raw_dir=str(tmp_path / "no_raw"),
        episodes_dir=str(tmp_path / "no_eps"),
        prefer_source="shard",  # exact behavior under test, not auto's fallback
        batch_size=4,
        context_len=16,
        frame_subsample=2,
        image_size=8,
        budget_gb=0.001,
        num_producers=1,
        min_goal_horizon=2,
        max_goal_horizon=4,
        min_fill=0.05,
    )
    loader = PipelinedGpuPretrainLoader(cfg)
    try:
        batch = next(loader)
    finally:
        loader.stop()

    ctx, goal, neg = batch["frames"], batch["goal"], batch["neg"]
    assert ctx.shape == (4, 9, 3, 8, 8)  # 9 ctx tokens (k=2 keeps last), then goal/neg slots
    assert goal.shape == neg.shape == (4, 3, 8, 8)
    # ramp: pixel value == native frame index in the loaded window, so the
    # goal gap vs the last ctx token (window[15]) is the horizon itself
    last_ctx = ctx[:, -1, 0, 0, 0].int()
    goal_gap = (goal[:, 0, 0, 0].int() - last_ctx).cpu()
    neg_gap = (neg[:, 0, 0, 0].int() - last_ctx).cpu()
    assert torch.all(goal_gap >= 2) and torch.all(goal_gap <= 4)
    assert torch.all(neg_gap > 4)
