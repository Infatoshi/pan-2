#!/usr/bin/env python3
"""Pretrain using GPU-resident pipelined loader (~10GB ring)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from pan2.config import load_config
from pan2.data.gpu_pipeline import PipelineConfig, PipelinedGpuPretrainLoader
from pan2.train.loop import build_state, save_ckpt, train_steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--budget-gb", type=float, default=10.0)
    p.add_argument("--producers", type=int, default=8)
    p.add_argument("--prefer-source", default="auto",
                   choices=["auto", "shard", "npy", "mp4", "pack"])
    p.add_argument("--raw-dir", default="/data/pan-2/raw")
    p.add_argument("--episodes-dir", default="/data/pan-2/episodes")
    p.add_argument("--pack-index", default="",
                   help="pack_index.npz from scripts/build_pack_index.py "
                        "(required when --prefer-source pack)")
    p.add_argument("--pack-minecraft-only", action="store_true",
                   help="restrict pack items to minecraft-flagged episodes")
    p.add_argument("--native-fps", type=float, default=None,
                   help="source fps for seek/horizon units (default 10 for "
                        "pack, 20 otherwise)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override config train.max_steps")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    cfg.train.stage = "pretrain"
    cfg.train.synthetic = False
    data_sub = cfg.model.frame_subsample
    cfg.model.frame_subsample = 1  # already_subsampled in ring
    state = build_state(cfg)
    cfg.model.frame_subsample = data_sub  # restore for pipeline config below

    native_fps = args.native_fps
    if native_fps is None:
        native_fps = 10.0 if args.prefer_source == "pack" else 20.0

    pcfg = PipelineConfig(
        raw_dir=args.raw_dir,
        episodes_dir=args.episodes_dir,
        batch_size=cfg.train.batch_size,
        context_len=cfg.model.context_len,
        frame_subsample=data_sub,
        image_size=cfg.model.image_size,
        budget_gb=args.budget_gb,
        num_producers=args.producers,
        prefer_source=args.prefer_source,
        device=cfg.train.device,
        min_goal_horizon=cfg.train.min_goal_horizon,
        max_goal_horizon=cfg.train.max_goal_horizon,
        native_fps=native_fps,
        pack_index=args.pack_index,
        pack_minecraft_only=args.pack_minecraft_only,
    )
    loader = PipelinedGpuPretrainLoader(pcfg)

    def gen():
        while True:
            yield next(loader)

    remaining = cfg.train.max_steps
    try:
        while remaining > 0:
            chunk = min(cfg.train.log_every, remaining)
            logs = train_steps(state, cfg, gen(), n_steps=chunk)
            st = loader.status()
            print(
                f"step={state.step} loss={logs[-1]['loss']:.4f} "
                f"ring={st['ready']}/{st['capacity']} fills={st['fills']} err={st['errors']}"
            )
            remaining -= chunk
            if state.step % cfg.train.ckpt_every == 0:
                save_ckpt(state, Path(cfg.train.ckpt_dir) / f"pretrain_step{state.step}.pt")
        save_ckpt(state, Path(cfg.train.ckpt_dir) / "pretrain_last.pt")
        print("pretrain done")
    finally:
        loader.stop()


if __name__ == "__main__":
    main()
