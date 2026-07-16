from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from pan2.config import Config
from pan2.models.policy import PanPolicy
from pan2.train.losses import action_loss, contrastive_loss
from pan2.train.speed import configure_cuda_fast_math


@dataclass
class TrainState:
    model: torch.nn.Module
    optim: torch.optim.Optimizer
    step: int = 0
    device: torch.device = torch.device("cpu")
    raw_model: PanPolicy | None = None


def build_state(cfg: Config) -> TrainState:
    configure_cuda_fast_math()
    # seed everything so A/B runs over different data variants share init+order
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = PanPolicy(cfg.model).to(device)
    raw_model = model if isinstance(model, PanPolicy) else None
    if cfg.train.compile and device.type == "cuda":
        # reduce-overhead good for steady train step; fullgraph=False for SDPA flexibility
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)  # type: ignore[assignment]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        fused=device.type == "cuda",
    )
    return TrainState(model=model, optim=optim, device=device, raw_model=raw_model)


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_steps(
    state: TrainState,
    cfg: Config,
    batches: Iterator[dict[str, torch.Tensor]],
    n_steps: int,
) -> list[dict[str, float]]:
    state.model.train()
    logs: list[dict[str, float]] = []
    for _ in range(n_steps):
        batch = next(batches)
        batch = {k: v.to(state.device, non_blocking=True) for k, v in batch.items()}
        neg = batch.get("neg") if cfg.train.hard_negatives else None
        state.optim.zero_grad(set_to_none=True)
        with _autocast(state.device, cfg.train.bf16):
            if cfg.train.stage == "pretrain":
                out = state.model(batch["frames"], batch["goal"], neg, return_actions=False)
                loss = contrastive_loss(out["contrastive_logits"])
                metrics = {"loss": float(loss.detach()), "stage": 0.0}
            elif cfg.train.stage == "posttrain":
                out = state.model(batch["frames"], batch["goal"], neg, return_actions=True)
                c_loss = contrastive_loss(out["contrastive_logits"])
                a_loss, a_metrics = action_loss(
                    out["discrete_logits"],
                    out["mouse_pred"],
                    batch["discrete"],
                    batch["mouse"],
                )
                loss = a_loss + 0.1 * c_loss
                metrics = {
                    "loss": float(loss.detach()),
                    "contrastive": float(c_loss.detach()),
                    **a_metrics,
                }
            else:
                raise ValueError(cfg.train.stage)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), cfg.train.grad_clip)
        state.optim.step()
        state.step += 1
        metrics["step"] = float(state.step)
        logs.append(metrics)
    return logs


def infinite_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _plain_ckpt_keys(sd: dict) -> dict:
    """Strip torch.compile's `._orig_mod.` segments so checkpoints are stable
    regardless of which submodule was compiled when they were written."""
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def _restore_ckpt_keys(model: torch.nn.Module, sd: dict) -> dict:
    """Re-insert `._orig_mod.` for every compiled submodule of `model` so a
    plain checkpoint loads into a build that has torch.compile enabled."""
    prefixes = [
        name
        for name, sub in model.named_modules()
        if name and hasattr(sub, "_orig_mod")
    ]
    for p in prefixes:
        sd = {
            (f"{p}._orig_mod.{k[len(p) + 1:]}"
             if k.startswith(f"{p}.") and not k.startswith(f"{p}._orig_mod.")
             else k): v
            for k, v in sd.items()
        }
    return sd


def save_ckpt(state: TrainState, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # unwrap compile if needed
    module = state.model
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod  # type: ignore[attr-defined]
    torch.save(
        {
            "step": state.step,
            "model": _plain_ckpt_keys(module.state_dict()),
            "optim": state.optim.state_dict(),
        },
        path,
    )


def load_ckpt(state: TrainState, path: str | Path, load_optim: bool = True) -> TrainState:
    ckpt = torch.load(path, map_location=state.device, weights_only=True)
    module = state.model
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod  # type: ignore[attr-defined]
    module.load_state_dict(_restore_ckpt_keys(module, _plain_ckpt_keys(ckpt["model"])))
    if load_optim and "optim" in ckpt:
        state.optim.load_state_dict(ckpt["optim"])
    state.step = int(ckpt.get("step", 0))
    return state
