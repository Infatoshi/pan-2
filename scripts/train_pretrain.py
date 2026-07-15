#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from pan2.config import load_config
from pan2.data.synthetic import SyntheticGoalDataset
from pan2.train.loop import build_state, infinite_loader, save_ckpt, train_steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pretrain_smoke.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.train.stage = "pretrain"
    state = build_state(cfg)

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
        print(f"step={state.step} loss={logs[-1]['loss']:.4f}")
        remaining -= chunk
        if state.step % cfg.train.ckpt_every == 0:
            save_ckpt(state, Path(cfg.train.ckpt_dir) / f"pretrain_step{state.step}.pt")
    save_ckpt(state, Path(cfg.train.ckpt_dir) / "pretrain_last.pt")
    print("pretrain done")


if __name__ == "__main__":
    main()
