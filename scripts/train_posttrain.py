#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from pan2.config import load_config
from pan2.data.synthetic import SyntheticGoalDataset
from pan2.train.loop import build_state, infinite_loader, load_ckpt, save_ckpt, train_steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/posttrain_smoke.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.train.stage = "posttrain"
    state = build_state(cfg)
    if cfg.train.pretrain_ckpt:
        load_ckpt(state, cfg.train.pretrain_ckpt, load_optim=False)

    if cfg.train.synthetic:
        ds = SyntheticGoalDataset(
            length=512,
            context_len=cfg.model.context_len,
            image_size=cfg.model.image_size,
            action_chunk=cfg.model.action_chunk,
            n_discrete=cfg.model.n_discrete,
        )
    else:
        from pan2.data.build import episode_dataset

        ds = episode_dataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )
    batches = infinite_loader(loader)
    remaining = cfg.train.max_steps
    while remaining > 0:
        chunk = min(cfg.train.log_every, remaining)
        logs = train_steps(state, cfg, batches, n_steps=chunk)
        m = logs[-1]
        extra = " ".join(
            f"{k}={m[k]:.4f}" for k in ("discrete_bce", "mouse_mse", "contrastive") if k in m
        )
        print(f"step={state.step} loss={m['loss']:.4f} {extra}")
        remaining -= chunk
        if state.step % cfg.train.ckpt_every == 0:
            save_ckpt(state, Path(cfg.train.ckpt_dir) / f"posttrain_step{state.step}.pt")
    save_ckpt(state, Path(cfg.train.ckpt_dir) / "posttrain_last.pt")
    print("posttrain done")


if __name__ == "__main__":
    main()
