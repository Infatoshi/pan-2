#!/usr/bin/env python3
"""Benchmark pipelined GPU ring loader + train step overlap."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pan2.config import ModelConfig
from pan2.data.gpu_pipeline import PipelineConfig, PipelinedGpuPretrainLoader
from pan2.models.policy import PanPolicy
from pan2.train.losses import contrastive_loss
from pan2.train.speed import configure_cuda_fast_math


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-gb", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--frame-subsample", type=int, default=8)
    p.add_argument("--producers", type=int, default=8)
    p.add_argument("--prefer-source", default="auto", choices=["auto", "npy", "mp4"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    configure_cuda_fast_math()
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}")

    pcfg = PipelineConfig(
        batch_size=args.batch_size,
        context_len=args.context_len,
        frame_subsample=args.frame_subsample,
        budget_gb=args.budget_gb,
        num_producers=args.producers,
        prefer_source=args.prefer_source,
        device="cuda",
    )
    t0 = time.perf_counter()
    loader = PipelinedGpuPretrainLoader(pcfg)
    print(f"fill_wait_s={time.perf_counter()-t0:.2f} status={loader.status()}")

    mcfg = ModelConfig(
        image_size=64,
        d_model=512,
        n_layers=8,
        n_heads=8,
        context_len=args.context_len,
        frame_subsample=1,  # already_subsampled in ring
        stem_channels=32,
        n_discrete=23,
    )
    model = PanPolicy(mcfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    model.train()

    def step():
        batch = next(loader)
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch["frames"], batch["goal"], return_actions=False)
            loss = contrastive_loss(out["contrastive_logits"])
        loss.backward()
        optim.step()
        return float(loss.detach())

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    # measure data-only (grab batch from ring)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = args.steps
    for _ in range(n):
        _ = next(loader)
    torch.cuda.synchronize()
    data_ms = 1000 * (time.perf_counter() - t0) / n

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = []
    for _ in range(n):
        losses.append(step())
    torch.cuda.synchronize()
    wall_ms = 1000 * (time.perf_counter() - t0) / n

    # pure compute with last batch shape
    batch = next(loader)
    for _ in range(5):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch["frames"], batch["goal"], return_actions=False)
            loss = contrastive_loss(out["contrastive_logits"])
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch["frames"], batch["goal"], return_actions=False)
            loss = contrastive_loss(out["contrastive_logits"])
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    gpu_ms = 1000 * (time.perf_counter() - t0) / n

    print("=== PIPELINE BENCH ===")
    print(f"ring_status={loader.status()}")
    print(f"batch_frames_shape={tuple(batch['frames'].shape)} dtype={batch['frames'].dtype}")
    print(f"data_only_ms={data_ms:.3f}  (GPU ring index; should be tiny when warm)")
    print(f"gpu_compute_only_ms={gpu_ms:.3f}")
    print(f"full_step_wall_ms={wall_ms:.3f}  steps/s={1000/wall_ms:.2f}")
    print(
        f"stall_vs_compute_ms={max(0.0, wall_ms-gpu_ms):.3f}  "
        f"({100*max(0.0, wall_ms-gpu_ms)/wall_ms:.1f}% of wall)"
    )
    print(f"last_loss={losses[-1]:.4f}")
    loader.stop()


if __name__ == "__main__":
    main()
