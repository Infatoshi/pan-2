"""Packed train-ready shards: the single ingest format for Stage A/B data.

Layout (one directory):

    shards/
      manifest.jsonl
        line 1: {"type":"header","version":1,"image_size":64,"act_dim":25,
                 "n_shards":N,"total_frames":T,"total_episodes":E}
        per episode: {"type":"segment","shard":0,"stem":"...","offset":F,
                      "n_frames":T_e,"has_act":true}
      shard-00000.frames.npy   uint8 (T_s, H, W, C), npy v1
      shard-00000.act.npy      float32 (T_s, act_dim)

Episodes never straddle shards (window sampling stays shard-local). Actions
share the frame offsets, so one index serves both stages. Numpy-memmap keeps
reads at page-cache speed; per-episode files or mp4 sources re-encode through
`scripts/build_shards.py`, which is also the funnel for new scraped video.
"""

from __future__ import annotations

import json
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

MANIFEST_NAME = "manifest.jsonl"
SHARD_VERSION = 1


class ShardWriter:
    """Pack uint8 frame episodes (+optional act rows) into shard files."""

    def __init__(
        self,
        out_dir: str | Path,
        image_size: int = 64,
        act_dim: int = 25,
        target_shard_bytes: int = 4 * 1024**3,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.act_dim = act_dim
        self.target_shard_bytes = target_shard_bytes
        self._frames_buf: list[np.ndarray] = []
        self._act_buf: list[np.ndarray] = []
        self._segments: list[dict] = []
        self._shard_idx = 0
        self._pending_bytes = 0
        self._total_frames = 0
        self._expect_act: bool | None = None
        self._closed = False

    def add_episode(self, frames: np.ndarray, act: np.ndarray | None, stem: str) -> None:
        if self._closed:
            raise RuntimeError("writer closed")
        has_act = act is not None
        if self._expect_act is None:
            self._expect_act = has_act
        elif has_act != self._expect_act:
            raise ValueError(
                f"{stem}: act presence must be uniform within one shard build "
                f"(first episode had act={self._expect_act})"
            )
        frames = np.ascontiguousarray(frames)
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError(f"{stem}: frames must be (T,H,W,3), got {frames.shape}")
        if frames.shape[1] != self.image_size or frames.shape[2] != self.image_size:
            raise ValueError(
                f"{stem}: expected {self.image_size}px, got {frames.shape[1]}x{frames.shape[2]}"
            )
        if frames.dtype != np.uint8:
            frames = frames.clip(0, 255).astype(np.uint8)
        t = int(frames.shape[0])
        if act is not None:
            act = np.ascontiguousarray(act, dtype=np.float32)
            if act.shape[0] != t:
                raise ValueError(f"{stem}: act length {act.shape[0]} != frames {t}")
        ep_bytes = frames.nbytes
        if self._pending_bytes > 0 and self._pending_bytes + ep_bytes > self.target_shard_bytes:
            self._flush()
        # shard-local offset: frames already buffered into THIS shard
        local_offset = sum(int(f.shape[0]) for f in self._frames_buf)
        self._segments.append(
            {
                "type": "segment",
                "shard": self._shard_idx,
                "stem": stem,
                "offset": local_offset,
                "n_frames": t,
                "has_act": act is not None,
            }
        )
        self._frames_buf.append(frames)
        if act is not None:
            self._act_buf.append(act)
        self._pending_bytes += ep_bytes
        self._total_frames += t

    def _flush(self) -> None:
        if not self._frames_buf:
            return
        frames = np.concatenate(self._frames_buf, axis=0)
        np.save(self.out_dir / f"shard-{self._shard_idx:05d}.frames.npy", frames)
        if self._act_buf:
            acts = np.concatenate(self._act_buf, axis=0)
            # segments without act are impossible in one shard build run
            # (act presence is uniform per source); assert consistency
            if acts.shape[0] != frames.shape[0]:
                raise RuntimeError(
                    f"shard {self._shard_idx}: act rows {acts.shape[0]} != frames"
                    f" {frames.shape[0]} (mixed act/no-act episodes in one shard)"
                )
            np.save(self.out_dir / f"shard-{self._shard_idx:05d}.act.npy", acts)
        self._frames_buf = []
        self._act_buf = []
        self._pending_bytes = 0
        self._shard_idx += 1

    def close(self) -> dict:
        self._flush()
        self._closed = True
        header = {
            "type": "header",
            "version": SHARD_VERSION,
            "image_size": self.image_size,
            "act_dim": self.act_dim,
            "n_shards": self._shard_idx,
            "total_frames": self._total_frames,
            "total_episodes": len(self._segments),
        }
        with open(self.out_dir / MANIFEST_NAME, "w") as f:
            f.write(json.dumps(header) + "\n")
            for seg in self._segments:
                f.write(json.dumps(seg) + "\n")
        return header


def load_manifest(shard_dir: str | Path) -> tuple[dict, list[dict]]:
    """-> (header, segments). Raises if the dir is not a shard build."""
    shard_dir = Path(shard_dir)
    path = shard_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no {MANIFEST_NAME} under {shard_dir}")
    header: dict | None = None
    segments: list[dict] = []
    for line in open(path):
        row = json.loads(line)
        if row.get("type") == "header":
            header = row
        elif row.get("type") == "segment":
            segments.append(row)
    if header is None:
        raise ValueError(f"{path}: missing header row")
    if header.get("version") != SHARD_VERSION:
        raise ValueError(f"{path}: version {header.get('version')} != {SHARD_VERSION}")
    return header, segments


class ShardDataset(Dataset):
    """Random-window reader over packed shards. Same sampling contract as
    VPTEpisodeDataset (see pan2.data.windowing)."""

    def __init__(
        self,
        shard_dir: str | Path,
        context_len: int = 128,
        action_chunk: int = 10,
        image_size: int = 64,
        max_episodes: int | None = None,
        keep_uint8: bool = True,
        min_goal_horizon: int = 20,
        max_goal_horizon: int = 300,
        windows_per_episode: int = 64,
    ):
        self.root = Path(shard_dir)
        self.header, self.segments = load_manifest(self.root)
        if max_episodes is not None:
            self.segments = self.segments[:max_episodes]
        if not self.segments:
            raise FileNotFoundError(f"no segments under {self.root}")
        if self.header["image_size"] != image_size:
            raise ValueError(f"shards are {self.header['image_size']}px, asked {image_size}")
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
        self.windows_per_episode = max(1, int(windows_per_episode))
        self.need = window_need(context_len, action_chunk, max_goal_horizon)
        self._seg_lens = np.array([s["n_frames"] for s in self.segments], dtype=np.int64)
        self._frames_cache: dict[int, np.ndarray] = {}
        self._act_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.segments) * self.windows_per_episode

    def _frames(self, shard: int) -> np.ndarray:
        if shard not in self._frames_cache:
            self._frames_cache[shard] = np.load(
                self.root / f"shard-{shard:05d}.frames.npy", mmap_mode="r"
            )
        return self._frames_cache[shard]

    def _acts(self, shard: int) -> np.ndarray:
        if shard not in self._act_cache:
            self._act_cache[shard] = np.load(
                self.root / f"shard-{shard:05d}.act.npy", mmap_mode="r"
            )
        return self._act_cache[shard]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seg = self.segments[idx % len(self.segments)]
        frames_all = self._frames(seg["shard"])
        off, t = int(seg["offset"]), int(seg["n_frames"])
        if t < self.need:
            frames_seg = np.asarray(frames_all[off : off + t])
            pad = np.repeat(frames_seg[-1:], self.need - t, axis=0)
            frames_seg = np.concatenate([frames_seg, pad], axis=0)
            if seg["has_act"]:
                a = np.asarray(self._acts(seg["shard"])[off : off + t])
                pad_a = np.repeat(a[-1:], self.need - t, axis=0)
                acts_seg = np.concatenate([a, pad_a], axis=0)
            else:
                acts_seg = np.zeros((self.need, self.header["act_dim"]), dtype=np.float32)
            t = self.need
        else:
            frames_seg = frames_all[off : off + t]  # memmap slice, cheap
            if seg["has_act"]:
                acts_seg = self._acts(seg["shard"])[off : off + t]
            else:
                acts_seg = np.zeros((t, self.header["act_dim"]), dtype=np.float32)
        w = sample_window(
            t,
            self.context_len,
            self.action_chunk,
            self.min_goal_horizon,
            self.max_goal_horizon,
        )
        window = np.ascontiguousarray(frames_seg[w.start : w.start + self.context_len])
        goal = np.ascontiguousarray(frames_seg[w.goal_idx])
        neg = np.ascontiguousarray(frames_seg[w.neg_idx])
        act_w = np.ascontiguousarray(acts_seg[w.act_start : w.act_start + self.action_chunk])
        discrete, mouse = split_actions(act_w)
        return {
            "frames": frames_to_tensor(window, self.keep_uint8),
            "goal": frames_to_tensor(goal[None, ...], self.keep_uint8)[0],
            "neg": frames_to_tensor(neg[None, ...], self.keep_uint8)[0],
            "discrete": discrete,
            "mouse": mouse,
        }
