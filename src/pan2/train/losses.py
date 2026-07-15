from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    b = logits.shape[0]
    labels = torch.arange(b, device=logits.device)
    return F.cross_entropy(logits, labels)


def action_loss(
    discrete_logits: torch.Tensor,
    mouse_pred: torch.Tensor,
    discrete_tgt: torch.Tensor,
    mouse_tgt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    h = min(discrete_logits.shape[1], discrete_tgt.shape[1])
    kd = min(discrete_logits.shape[2], discrete_tgt.shape[2])
    km = min(mouse_pred.shape[2], mouse_tgt.shape[2])
    d_loss = F.binary_cross_entropy_with_logits(
        discrete_logits[:, :h, :kd], discrete_tgt[:, :h, :kd].clamp(0, 1)
    )
    m_loss = F.mse_loss(mouse_pred[:, :h, :km], mouse_tgt[:, :h, :km])
    total = d_loss + m_loss
    return total, {
        "discrete_bce": float(d_loss.detach()),
        "mouse_mse": float(m_loss.detach()),
    }
