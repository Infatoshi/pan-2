from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pan2.actions import MOUSE_DIM


class VPTEpisodeDataset(Dataset):
    """Read preprocessed VPT episodes: `*.img.npy` + matching `*.act.npy`.

    Returns frames/goal as uint8 NCHW for cheap H2D; normalize/resize on GPU.

    Goal sampling (fixed 2026-07-15): the goal is a FUTURE frame strictly past
    the context window: goal_idx = start + context_len - 1 + horizon, with
    horizon ~ U(min_goal_horizon, max_goal_horizon) native frames. The previous
    version used the last context frame as the goal, which collapses Stage-A
    contrastive pretraining to duplicate detection.

    Hard negative: each sample also returns `neg`, a frame from the SAME
    episode strictly beyond the goal window (idx >= context_end + max_horizon
    + 1). Scene statistics persist within an episode, so this distractor
    defeats trivial scene-ID matching that in-batch (cross-episode) negatives
    allow.

    Windows are returned FULL-RATE; temporal subsampling happens once, either
    on-GPU in PanPolicy (this path) or at ring fill time (gpu_pipeline path).
    Passing subsampled windows through a subsampling model used to double-apply
    the stride.
    """

    def __init__(
        self,
        root: str | Path,
        context_len: int = 128,
        action_chunk: int = 10,
        image_size: int = 64,
        max_episodes: int | None = None,
        keep_uint8: bool = True,
        min_goal_horizon: int = 20,
        max_goal_horizon: int = 300,
        windows_per_episode: int = 64,
    ):
        self.root = Path(root)
        self.context_len = context_len
        self.action_chunk = action_chunk
        self.image_size = image_size
        self.keep_uint8 = keep_uint8
        if min_goal_horizon < 1:
            raise ValueError("min_goal_horizon must be >= 1 (goal strictly after context)")
        if max_goal_horizon < min_goal_horizon:
            raise ValueError("max_goal_horizon must be >= min_goal_horizon")
        self.min_goal_horizon = min_goal_horizon
        self.max_goal_horizon = max_goal_horizon
        # epoch = len(pairs) * windows_per_episode random-window draws. Named
        # knob (not a magic multiplier) so tiny datasets still exceed any
        # sane batch_size; getitem maps each draw to an episode by modulo.
        self.windows_per_episode = max(1, int(windows_per_episode))
        imgs = sorted(self.root.glob("*.img.npy"))
        if max_episodes is not None:
            imgs = imgs[:max_episodes]
        self.pairs: list[tuple[Path, Path]] = []
        for img in imgs:
            act = img.with_name(img.name.replace(".img.npy", ".act.npy"))
            if act.exists():
                self.pairs.append((img, act))
        if not self.pairs:
            raise FileNotFoundError(f"no img/act pairs under {self.root}")

    def __len__(self) -> int:
        return len(self.pairs) * self.windows_per_episode

    def _need(self) -> int:
        # context + largest reach of action chunk, goal, or hard negative
        stretch = max(self.action_chunk, 2 * self.max_goal_horizon)
        return self.context_len - 1 + stretch + 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, act_path = self.pairs[idx % len(self.pairs)]
        frames = np.load(img_path, mmap_mode="r")
        acts = np.load(act_path, mmap_mode="r")
        t = int(frames.shape[0])
        need = self._need()
        if t < need:
            pad_f = np.repeat(frames[-1:], need - t, axis=0)
            pad_a = np.repeat(acts[-1:], need - t, axis=0)
            frames_np = np.concatenate([np.asarray(frames), pad_f], axis=0)
            acts_np = np.concatenate([np.asarray(acts), pad_a], axis=0)
            t = need
        else:
            frames_np = frames
            acts_np = acts
        start = int(np.random.randint(0, t - need + 1))
        # Copy slice so tensor is writable and independent of mmap.
        window = np.ascontiguousarray(frames_np[start : start + self.context_len])
        horizon = int(np.random.randint(self.min_goal_horizon, self.max_goal_horizon + 1))
        goal_idx = start + self.context_len - 1 + horizon
        goal = np.ascontiguousarray(frames_np[goal_idx])
        # hard negative: same episode, strictly beyond the goal window
        neg_off = int(np.random.randint(1, self.max_goal_horizon + 1))
        neg_idx = start + self.context_len - 1 + self.max_goal_horizon + neg_off
        neg = np.ascontiguousarray(frames_np[neg_idx])
        act_w = np.ascontiguousarray(
            acts_np[
                start + self.context_len - 1 : start + self.context_len - 1 + self.action_chunk
            ]
        )
        frames_t = self._frames_to_tensor(window)
        goal_t = self._frames_to_tensor(goal[None, ...])[0]
        neg_t = self._frames_to_tensor(neg[None, ...])[0]
        discrete, mouse = self._split_actions(act_w)
        return {
            "frames": frames_t,
            "goal": goal_t,
            "neg": neg_t,
            "discrete": discrete,
            "mouse": mouse,
        }

    def _frames_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """NHWC numpy -> NCHW torch. uint8 by default (no CPU float/resize)."""
        x = np.ascontiguousarray(arr)
        if x.ndim == 3:
            x = x[None, ...]
        # NHWC -> NCHW
        x = np.transpose(x, (0, 3, 1, 2))
        if self.keep_uint8:
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

    def _split_actions(self, act: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
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
