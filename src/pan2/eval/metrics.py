from __future__ import annotations

import torch


def contrastive_accuracy(logits: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return float((preds == labels).float().mean().item())
