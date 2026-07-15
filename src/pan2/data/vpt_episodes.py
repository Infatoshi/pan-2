from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pan2.data.windowing import (
    frames_to_tensor,
    sample_window,
    split_actions,
    window_need,
)


class VPTEpisodeDataset(Dataset):
    """Read preprocessed VPT episodes: `*.img.npy` + matching `*.act.npy`.

    Returns frames/goal as uint8 NCHW for cheap H2D; normalize/resize on GPU.
    Goal/neg/action sampling contract: see pan2.data.windowing (strictly
    future goals, same-episode hard negatives, full-rate windows; temporal
    subsampling happens once, in PanPolicy or at ring fill time).
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
        return window_need(self.context_len, self.action_chunk, self.max_goal_horizon)

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
        w = sample_window(
            t,
            self.context_len,
            self.action_chunk,
            self.min_goal_horizon,
            self.max_goal_horizon,
        )
        # Copy slices so tensors are writable and independent of mmap.
        window = np.ascontiguousarray(frames_np[w.start : w.start + self.context_len])
        goal = np.ascontiguousarray(frames_np[w.goal_idx])
        neg = np.ascontiguousarray(frames_np[w.neg_idx])
        act_w = np.ascontiguousarray(acts_np[w.act_start : w.act_start + self.action_chunk])
        frames_t = frames_to_tensor(window, self.keep_uint8)
        goal_t = frames_to_tensor(goal[None, ...], self.keep_uint8)[0]
        neg_t = frames_to_tensor(neg[None, ...], self.keep_uint8)[0]
        discrete, mouse = split_actions(act_w)
        return {
            "frames": frames_t,
            "goal": goal_t,
            "neg": neg_t,
            "discrete": discrete,
            "mouse": mouse,
        }
