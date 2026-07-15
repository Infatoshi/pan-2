"""Regression tests for goal sampling (goal must be strictly after context)."""

import numpy as np

from pan2.data.vpt_episodes import VPTEpisodeDataset


def _write_ramp_episode(root, stem="ep0", t=240):
    # frame i is filled with value i (kept < 250 so identity survives uint8)
    img = np.zeros((t, 4, 4, 3), dtype=np.uint8)
    for i in range(t):
        img[i] = i
    act = np.zeros((t, 25), dtype=np.float32)
    np.save(root / f"{stem}.img.npy", img)
    np.save(root / f"{stem}.act.npy", act)


def test_goal_is_future_frame(tmp_path):
    context_len, min_h, max_h = 32, 4, 40
    _write_ramp_episode(tmp_path)
    ds = VPTEpisodeDataset(
        tmp_path,
        context_len=context_len,
        action_chunk=10,
        image_size=4,
        min_goal_horizon=min_h,
        max_goal_horizon=max_h,
    )
    assert len(ds) == ds.windows_per_episode  # 1 episode x windows_per_episode draws
    for _ in range(25):
        item = ds[0]
        last_ctx = int(item["frames"][-1, 0, 0, 0])
        goal_val = int(item["goal"][0, 0, 0])
        # ramp increases with frame index (250 > T so no wraparound)
        gap = goal_val - last_ctx
        assert min_h <= gap <= max_h, f"goal gap {gap} outside [{min_h}, {max_h}]"
        neg_val = int(item["neg"][0, 0, 0])
        neg_gap = neg_val - last_ctx
        assert neg_gap > max_h, f"neg gap {neg_gap} not beyond goal window ({max_h})"


def test_actions_still_full_rate_at_last_frame(tmp_path):
    context_len, chunk = 32, 10
    _write_ramp_episode(tmp_path)
    ds = VPTEpisodeDataset(
        tmp_path,
        context_len=context_len,
        action_chunk=chunk,
        image_size=4,
        min_goal_horizon=1,
        max_goal_horizon=20,
    )
    item = ds[0]
    assert item["discrete"].shape == (chunk, 23)
    assert item["mouse"].shape == (chunk, 2)
    assert item["frames"].shape[0] == context_len  # full-rate window, no subsample here
