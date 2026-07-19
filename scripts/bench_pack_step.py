#!/usr/bin/env python3
"""Steady-state step-time + batch scaling bench for the pack pipeline.

Mirrors production exactly: pretrain_pack.yaml model/train config, pack
loader, K=4, FusedAdamW fused clip, bf16 autocast, optional
PAN2_TEMPORAL_COMPILE_MODE from env. Times with in-process perf_counter
over steps AFTER warmup, so numbers are uncontaminated by startup/JIT.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pan2.config import load_config
from pan2.data.gpu_pipeline import PipelineConfig, PipelinedGpuPretrainLoader
from pan2.train.loop import build_state, train_steps


def _kernel_table(prof, steps: int, top: int) -> None:
    """Top kernels by self device time, with launch counts and per-step ms."""
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for ev in prof.key_averages():
        if ev.self_device_time_total <= 0:
            continue
        agg[ev.key][0] += ev.self_device_time_total / 1000.0  # us -> ms
        agg[ev.key][1] += ev.count
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]
    total = sum(v[0] for v in agg.values())
    print(f"gpu_self_ms/step={total / steps:.2f} over {steps} profiled steps")
    print(f"{'ms/step':>8} {'calls/step':>10}  {'%gpu':>5}  name")
    for name, (ms, cnt) in rows:
        print(f"{ms / steps:8.2f} {cnt / steps:10.1f} {100 * ms / total:5.1f}  {name[:90]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pretrain_pack.yaml")
    p.add_argument("--pack-index", default="data/crawl/pack/pack_index.npz")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--producers", type=int, default=12)
    p.add_argument("--budget-gb", type=float, default=10.0)
    p.add_argument("--profile-steps", type=int, default=0,
                   help="profile N post-warmup steps and print the kernel table")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.train.stage = "pretrain"
    cfg.train.synthetic = False
    cfg.train.batch_size = args.batch_size
    # pack applies stride at the data layer already (frame_subsample 1)
    state = build_state(cfg)

    pcfg = PipelineConfig(
        batch_size=cfg.train.batch_size,
        context_len=cfg.model.context_len,
        frame_subsample=cfg.model.frame_subsample,
        image_size=cfg.model.image_size,
        budget_gb=args.budget_gb,
        num_producers=args.producers,
        prefer_source="pack",
        device=cfg.train.device,
        min_goal_horizon=cfg.train.min_goal_horizon,
        max_goal_horizon=cfg.train.max_goal_horizon,
        n_hard_negatives=cfg.train.n_hard_negatives,
        native_fps=10.0,
        pack_index=args.pack_index,
    )
    loader = PipelinedGpuPretrainLoader(pcfg)

    def gen():
        while True:
            yield next(loader)

    try:
        train_steps(state, cfg, gen(), n_steps=args.warmup)
        if args.profile_steps > 0:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                train_steps(state, cfg, gen(), n_steps=args.profile_steps)
            torch.cuda.synchronize()
            _kernel_table(prof, args.profile_steps, top=20)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_steps(state, cfg, gen(), n_steps=args.steps)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    finally:
        loader.stop()

    ms = 1000 * dt / args.steps
    clips_s = args.steps * args.batch_size / dt
    # each clip covers context_len + up to max_goal_horizon native frames;
    # report the hours-of-video/s the way DEVLOG does (ctx+goal seconds at 10fps)
    secs_per_clip = (cfg.model.context_len + 1) / 10.0
    hv_s = clips_s * secs_per_clip / 3600
    print(
        f"RESULT bs={args.batch_size} producers={args.producers} "
        f"{ms:.2f} ms/step {clips_s:.0f} clips/s {hv_s:.2f} h-video/s "
        f"({args.warmup} warmup excluded)"
    )


if __name__ == "__main__":
    main()
