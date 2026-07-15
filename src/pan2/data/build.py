"""Single place that picks the episode dataset implementation.

If data_dir holds a shard build (manifest.jsonl) -> ShardDataset, else assume
per-episode *.img.npy pairs -> VPTEpisodeDataset. Same sampling contract and
constructor params either way, so train scripts never branch on format.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from pan2.config import Config
from pan2.data.shards import MANIFEST_NAME, ShardDataset
from pan2.data.vpt_episodes import VPTEpisodeDataset


def episode_dataset(cfg: Config) -> Dataset:
    root = Path(cfg.train.data_dir)
    cls = ShardDataset if (root / MANIFEST_NAME).exists() else VPTEpisodeDataset
    return cls(
        root,
        context_len=cfg.model.context_len,
        action_chunk=cfg.model.action_chunk,
        image_size=cfg.model.image_size,
        max_episodes=cfg.train.max_episodes,
        keep_uint8=cfg.train.keep_uint8,
        min_goal_horizon=cfg.train.min_goal_horizon,
        max_goal_horizon=cfg.train.max_goal_horizon,
    )
