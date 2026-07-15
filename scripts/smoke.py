#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pan2.config import load_config
from pan2.data.synthetic import synthetic_batch
from pan2.eval.metrics import contrastive_accuracy
from pan2.train.loop import build_state, train_steps


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for cfg_path, stage in [
        ("configs/pretrain_smoke.yaml", "pretrain"),
        ("configs/posttrain_smoke.yaml", "posttrain"),
    ]:
        cfg = load_config(cfg_path)
        cfg.train.device = device
        cfg.train.stage = stage
        state = build_state(cfg)

        def gen():
            while True:
                yield synthetic_batch(
                    cfg.train.batch_size,
                    cfg.model.context_len,
                    cfg.model.image_size,
                    cfg.model.action_chunk,
                    cfg.model.n_discrete,
                    cfg.model.mouse_dim,
                    uint8=True,
                )

        logs = train_steps(state, cfg, gen(), n_steps=3)
        batch = synthetic_batch(
            cfg.train.batch_size,
            cfg.model.context_len,
            cfg.model.image_size,
            cfg.model.action_chunk,
            cfg.model.n_discrete,
            device=state.device,
            uint8=True,
        )
        state.model.eval()
        with torch.no_grad():
            out = state.model(
                batch["frames"], batch["goal"], return_actions=(stage == "posttrain")
            )
            acc = contrastive_accuracy(out["contrastive_logits"])
        print(f"[{stage}] steps_ok last_loss={logs[-1]['loss']:.4f} contrastive_acc={acc:.3f}")
    print("smoke ok")


if __name__ == "__main__":
    main()
