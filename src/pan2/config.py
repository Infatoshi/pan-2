from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    image_size: int = 64
    in_channels: int = 3
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    context_len: int = 128
    action_chunk: int = 10
    n_discrete: int = 23
    mouse_dim: int = 2
    backbone: str = "transformer"
    # encode every k-th frame (always keeps last). Data is 20Hz native:
    # k=2 -> ~10fps tokens (Pan rate); k>=8 destroys action-timing relevance.
    frame_subsample: int = 2
    stem_channels: int = 32


@dataclass
class TrainConfig:
    stage: str = "pretrain"
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    log_every: int = 20
    ckpt_every: int = 200
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    bf16: bool = True
    synthetic: bool = False
    data_dir: str = "/data/pan-2/episodes"
    ckpt_dir: str = "data/checkpoints"
    resume: str | None = None
    pretrain_ckpt: str | None = None
    max_episodes: int | None = None  # cap dataset size (overfit sanity)
    temperature: float = 0.07
    compile: bool = False
    keep_uint8: bool = True
    # hindsight goal horizon in native (20Hz) frames, strictly past context end
    min_goal_horizon: int = 20
    max_goal_horizon: int = 300
    # same-episode hard negative as extra contrastive column (defeats scene-ID
    # shortcut). Disable for ablation.
    hard_negatives: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    model = ModelConfig(**(raw.get("model") or {}))
    train = TrainConfig(**(raw.get("train") or {}))
    return Config(model=model, train=train)
