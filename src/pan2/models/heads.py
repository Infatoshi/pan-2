from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalValueHead(nn.Module):
    def __init__(self, d_model: int, proj_dim: int | None = None):
        super().__init__()
        proj_dim = proj_dim or d_model
        self.state_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.goal_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def encode_state(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.state_proj(h), dim=-1)

    def encode_goal(self, g: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.goal_proj(g), dim=-1)

    def logits(
        self,
        state_tok: torch.Tensor,
        goal_tok: torch.Tensor,
        neg_tok: torch.Tensor | None = None,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """[B,D] state/goal (+optional [B,D] same-episode neg) -> [B,B(+1)] logits.

        Row i's positive is column i. The optional final column is that row's
        own hard negative, which defeats scene-ID shortcuts that cross-episode
        in-batch negatives cannot.
        """
        s = self.encode_state(state_tok)
        g = self.encode_goal(goal_tok)
        out = (s @ g.T) / temperature
        if neg_tok is not None:
            n = self.encode_goal(neg_tok)
            own_neg = (s * n).sum(dim=-1, keepdim=True) / temperature
            out = torch.cat([out, own_neg], dim=1)
        return out


class ActionChunkHead(nn.Module):
    def __init__(self, d_model: int, n_discrete: int, mouse_dim: int, chunk: int):
        super().__init__()
        self.chunk = chunk
        self.n_discrete = n_discrete
        self.mouse_dim = mouse_dim
        out = chunk * (n_discrete + mouse_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.net(h)
        b = y.shape[0]
        y = y.view(b, self.chunk, self.n_discrete + self.mouse_dim)
        disc = y[..., : self.n_discrete]
        mouse = y[..., self.n_discrete :]
        return disc, mouse
