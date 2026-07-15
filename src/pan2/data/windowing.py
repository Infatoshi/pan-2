"""Shared window-sampling + tensor conversion for episode-shaped data.

Used by both VPTEpisodeDataset (per-episode npy) and ShardDataset (packed
shards) so goal/neg/horizon semantics can never drift between loaders.

Contract (fixed 2026-07-15):
- goal: strictly-future frame, goal_idx = start + context_len - 1 + horizon,
  horizon ~ U(min_goal_horizon, max_goal_horizon) native frames
- neg: same-episode frame strictly past the goal window,
  neg_idx = start + context_len - 1 + max_goal_horizon + U(1, max_goal_horizon)
- actions: full-rate chunk starting at the last context frame
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pan2.actions import MOUSE_DIM


def window_need(context_len: int, action_chunk: int, max_goal_horizon: int) -> int:
    """Frames a sampling window must span: context + furthest reach."""
    stretch = max(action_chunk, 2 * max_goal_horizon)
    return context_len - 1 + stretch + 1


@dataclass
class WindowSample:
    start: int
    goal_idx: int
    neg_idx: int
    act_start: int


def sample_window(
    n_frames: int,
    context_len: int,
    action_chunk: int,
    min_goal_horizon: int,
    max_goal_horizon: int,
    randint=np.random.randint,
) -> WindowSample:
    """Sample window params inside an episode of n_frames (already padded to need)."""
    need = window_need(context_len, action_chunk, max_goal_horizon)
    start = int(randint(0, n_frames - need + 1))
    horizon = int(randint(min_goal_horizon, max_goal_horizon + 1))
    neg_off = int(randint(1, max_goal_horizon + 1))
    return WindowSample(
        start=start,
        goal_idx=start + context_len - 1 + horizon,
        neg_idx=start + context_len - 1 + max_goal_horizon + neg_off,
        act_start=start + context_len - 1,
    )


def frames_to_tensor(arr: np.ndarray, keep_uint8: bool) -> torch.Tensor:
    """NHWC numpy -> NCHW torch. uint8 by default (no CPU float/resize)."""
    x = np.ascontiguousarray(arr)
    if x.ndim == 3:
        x = x[None, ...]
    # NHWC -> NCHW
    x = np.transpose(x, (0, 3, 1, 2))
    if keep_uint8:
        if x.dtype != np.uint8:
            if x.max() <= 1.0:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        return torch.from_numpy(x.copy())  # uint8 NCHW, writable
    # legacy float path (CPU normalize; still no resize — GPU does it)
    t = torch.from_numpy(x.copy())
    if t.dtype == torch.uint8:
        t = t.float().div_(255.0)
    else:
        t = t.float()
        if float(t.max()) > 1.5:
            t = t / 255.0
    return t


def split_actions(act: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    # Layout measured from data (see pan2.actions): cols 0-22 binary
    # buttons, cols 23-24 camera dx/dy in 0.1 steps over [-1, 1].
    a = torch.from_numpy(np.array(act, copy=True, order="C")).float()
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if a.shape[-1] >= 3:
        mouse = a[:, -MOUSE_DIM:]
        disc = a[:, :-MOUSE_DIM]
    else:
        disc = a
        mouse = torch.zeros(a.shape[0], 2)
    return disc, mouse
