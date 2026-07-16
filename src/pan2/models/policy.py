from __future__ import annotations

import torch
import torch.nn as nn

from pan2.config import ModelConfig
from pan2.models.encoder import FrameEncoder
from pan2.models.heads import ActionChunkHead, GoalValueHead
from pan2.models.preprocess import prepare_images
from pan2.models.temporal import build_temporal


class PanPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.frame_subsample = max(1, int(getattr(cfg, "frame_subsample", 1)))
        self.encoder = FrameEncoder(
            d_model=cfg.d_model,
            in_channels=cfg.in_channels,
            image_size=cfg.image_size,
            stem_channels=getattr(cfg, "stem_channels", 32),
        )
        # pos table sized for worst case (no subsample) + goal
        self.temporal = build_temporal(
            cfg.backbone,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            max_len=cfg.context_len + 1,
        )
        self.value_head = GoalValueHead(cfg.d_model)
        self.action_head = ActionChunkHead(
            d_model=cfg.d_model,
            n_discrete=cfg.n_discrete,
            mouse_dim=cfg.mouse_dim,
            chunk=cfg.action_chunk,
        )

    def _subsample_time(self, frames: torch.Tensor) -> torch.Tensor:
        """Subsample along T before cast/encode. Always keeps last frame."""
        k = self.frame_subsample
        if k <= 1:
            return frames
        t = frames.shape[1]
        idx = torch.arange(0, t, k, device=frames.device)
        if int(idx[-1].item()) != t - 1:
            idx = torch.cat([idx, idx.new_tensor([t - 1])])
        return frames.index_select(1, idx)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W] prepared float, already time-subsampled
        b, t, c, h, w = frames.shape
        flat = frames.flatten(0, 1)
        tok = self.encoder(flat)
        return tok.view(b, t, -1)

    def forward(
        self,
        frames: torch.Tensor,
        goal: torch.Tensor,
        neg: torch.Tensor | None = None,
        *,
        return_actions: bool = False,
    ) -> dict[str, torch.Tensor]:
        # subsample first (cheap on uint8) so cast/encode see T/k frames only
        frames = self._subsample_time(frames)
        frames = prepare_images(frames, self.cfg.image_size)
        goal = prepare_images(goal, self.cfg.image_size)

        frame_tok = self.encode_frames(frames)
        goal_tok = self.encoder(goal).unsqueeze(1)
        tokens = torch.cat([frame_tok, goal_tok], dim=1)
        hidden = self.temporal(tokens)
        state = hidden[:, -2, :]
        cond = hidden[:, -1, :]
        out: dict[str, torch.Tensor] = {
            "frame_tok": frame_tok,
            "goal_tok": goal_tok.squeeze(1),
            "cond": cond,
            "state": state,
        }
        neg_tok = None
        if neg is not None:
            neg = prepare_images(neg, self.cfg.image_size)
            neg_tok = self.encoder(neg)
        out["contrastive_logits"] = self.value_head.logits(cond, out["goal_tok"], neg_tok)
        if return_actions:
            disc, mouse = self.action_head(cond)
            out["discrete_logits"] = disc
            out["mouse_pred"] = mouse
        return out
