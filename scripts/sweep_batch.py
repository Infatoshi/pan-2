#!/usr/bin/env python3
"""Sweep train batch size on warm GPU ring; report bottleneck at max stable B."""
from __future__ import annotations

import argparse
import gc
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


def mem_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2)


def try_batch(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loader: PipelinedGpuPretrainLoader,
    batch_size: int,
    steps: int,
    warmup: int,
) -> dict:
    device = next(model.parameters()).device
    # temporarily override loader batch size by sampling multiple times if needed
    # easier: rebuild is expensive; instead gather B slots manually from ring
    ring = loader.ring
    rng = loader.rng
    t_sub = ring.t_sub

    def get_batch():
        # wait for enough ready
        for _ in range(10000):
            slots = ring.sample_slots(batch_size, rng)
            if slots:
                break
            time.sleep(0.002)
        else:
            raise RuntimeError(f"not enough ready slots for B={batch_size}")
        slot_t = torch.tensor(slots, device=device, dtype=torch.long)
        frames = ring.frames.index_select(0, slot_t)
        goal_idx = torch.randint(0, t_sub, (batch_size,), device=device)
        b_idx = torch.arange(batch_size, device=device)
        goal = frames[b_idx, goal_idx]
        return {"frames": frames, "goal": goal}

    def step(batch):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch["frames"], batch["goal"], return_actions=False)
            loss = contrastive_loss(out["contrastive_logits"])
        loss.backward()
        optim.step()
        return float(loss.detach())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # warmup
    for _ in range(warmup):
        step(get_batch())
    torch.cuda.synchronize()

    # data-only
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        _ = get_batch()
    torch.cuda.synchronize()
    data_ms = 1000 * (time.perf_counter() - t0) / steps

    # full wall
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = None
    for _ in range(steps):
        loss = step(get_batch())
    torch.cuda.synchronize()
    wall_ms = 1000 * (time.perf_counter() - t0) / steps

    # compute-only (fixed batch already on GPU)
    fixed = get_batch()
    for _ in range(3):
        step(fixed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step(fixed)
    torch.cuda.synchronize()
    gpu_ms = 1000 * (time.perf_counter() - t0) / steps

    peak = mem_mb()
    frames_per_step = batch_size * t_sub  # encoded frames roughly
    return {
        "batch_size": batch_size,
        "data_ms": data_ms,
        "gpu_ms": gpu_ms,
        "wall_ms": wall_ms,
        "steps_s": 1000.0 / wall_ms,
        "clips_s": batch_size * 1000.0 / wall_ms,
        "frames_s": frames_per_step * 1000.0 / wall_ms,
        "peak_mem_mb": peak,
        "stall_ms": max(0.0, wall_ms - gpu_ms),
        "stall_pct": 100.0 * max(0.0, wall_ms - gpu_ms) / wall_ms,
        "data_pct": 100.0 * data_ms / wall_ms,
        "gpu_pct": 100.0 * gpu_ms / wall_ms,
        "loss": loss,
        "ok": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--budget-gb",
        type=float,
        default=3.0,
        help="ring size; leave rest for activations",
    )
    p.add_argument("--producers", type=int, default=8)
    p.add_argument("--batches", default="32,64,128,256,384,512,768,1024")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--profile-top", type=int, default=0, help="if >0, profile largest OK batch")
    args = p.parse_args()

    configure_cuda_fast_math()
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"vram_free_gb={free/1e9:.2f} total_gb={total/1e9:.2f}")

    pcfg = PipelineConfig(
        batch_size=32,
        context_len=128,
        frame_subsample=2,
        image_size=64,
        budget_gb=args.budget_gb,
        num_producers=args.producers,
        prefer_source="auto",
        device="cuda",
        min_fill=0.1,
    )
    loader = PipelinedGpuPretrainLoader(pcfg)
    print(f"ring={loader.status()}")

    mcfg = ModelConfig(
        image_size=64,
        d_model=512,
        n_layers=8,
        n_heads=8,
        context_len=128,
        frame_subsample=1,  # ring already subsampled
        stem_channels=32,
        n_discrete=23,
    )
    model = PanPolicy(mcfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    model.train()
    print(f"model_params_M={sum(p.numel() for p in model.parameters())/1e6:.2f}")

    sizes = [int(x) for x in args.batches.split(",") if x.strip()]
    results = []
    best = None
    for b in sizes:
        # need enough ready slots
        if loader.ring.num_ready() < b:
            print(f"B={b}: skip (ready={loader.ring.num_ready()} < batch)")
            continue
        try:
            # pause? leave producers running; ring already large
            r = try_batch(model, optim, loader, b, args.steps, args.warmup)
            results.append(r)
            print(
                f"B={b:4d}  wall={r['wall_ms']:7.2f}ms  gpu={r['gpu_ms']:7.2f}ms  "
                f"data={r['data_ms']:6.2f}ms  stall={r['stall_pct']:5.1f}%  "
                f"clips/s={r['clips_s']:8.1f}  frames/s={r['frames_s']:9.0f}  "
                f"peak_mem={r['peak_mem_mb']:7.0f}MB  loss={r['loss']:.3f}"
            )
            best = r
        except torch.cuda.OutOfMemoryError:
            print(f"B={b:4d}  OOM")
            torch.cuda.empty_cache()
            gc.collect()
            break
        except Exception as e:
            print(f"B={b:4d}  FAIL {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            break

    print("\n=== SWEEP SUMMARY ===")
    if not results:
        print("no successful batches")
        loader.stop()
        return

    # best by clips/s
    by_clips = max(results, key=lambda r: r["clips_s"])
    by_wall = min(results, key=lambda r: r["wall_ms"])
    largest = results[-1]
    print(
        f"largest_ok_B={largest['batch_size']}  "
        f"best_clips_s_B={by_clips['batch_size']} ({by_clips['clips_s']:.1f}/s)  "
        f"best_wall_B={by_wall['batch_size']} ({by_wall['wall_ms']:.2f}ms)"
    )
    print(
        f"at largest B={largest['batch_size']}: "
        f"gpu_pct={largest['gpu_pct']:.1f}% data_pct={largest['data_pct']:.1f}% "
        f"stall_pct={largest['stall_pct']:.1f}%"
    )

    # bottleneck call for largest
    r = largest
    if r["gpu_pct"] >= 85:
        bot = "GPU_COMPUTE"
    elif r["data_pct"] >= 20:
        bot = "DATA_RING_SAMPLE"
    elif r["stall_pct"] >= 20:
        bot = "STALL_OVERHEAD (launch/sync/fragmentation/producers)"
    else:
        bot = "MIXED"
    print(f"BOTTLENECK_AT_LARGEST={bot}")

    # optional quick profiler on largest
    if args.profile_top > 0 and best is not None:
        from collections import defaultdict

        from torch.profiler import ProfilerActivity, profile, schedule

        B = largest["batch_size"]
        print(f"\n=== PROFILE largest B={B} ===")

        def get_batch():
            for _ in range(10000):
                slots = loader.ring.sample_slots(B, loader.rng)
                if slots:
                    break
                time.sleep(0.002)
            slot_t = torch.tensor(slots, device=device, dtype=torch.long)
            frames = loader.ring.frames.index_select(0, slot_t)
            goal_idx = torch.randint(0, loader.ring.t_sub, (B,), device=device)
            b_idx = torch.arange(B, device=device)
            return {"frames": frames, "goal": frames[b_idx, goal_idx]}

        def train_step():
            batch = get_batch()
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch["frames"], batch["goal"], return_actions=False)
                loss = contrastive_loss(out["contrastive_logits"])
            loss.backward()
            optim.step()
            return loss

        for _ in range(5):
            train_step()
        torch.cuda.synchronize()

        wait, warmup, active = 1, 5, 15
        sched = schedule(wait=wait, warmup=warmup, active=active, repeat=1)
        walls = []
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=False,
        ) as prof:
            for i in range(wait + warmup + active):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                train_step()
                e.record()
                torch.cuda.synchronize()
                if i >= wait + warmup:
                    walls.append(s.elapsed_time(e))
                prof.step()
        avg_wall = sum(walls) / len(walls)
        print(f"avg_wall_ms={avg_wall:.3f}")

        # top self cuda excluding user ranges
        exclude = {"forward", "backward", "loss", "optim", "train_step"}
        kernel_us: dict[str, float] = defaultdict(float)
        for evt in prof.key_averages():
            if evt.key in exclude or str(evt.key).startswith("Profiler"):
                continue
            us = getattr(evt, "self_device_time_total", None)
            if us is None:
                us = getattr(evt, "self_cuda_time_total", 0) or 0
            if us and us > 0:
                kernel_us[evt.key] += float(us)
        total = sum(kernel_us.values()) or 1.0
        ranked = sorted(kernel_us.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print(f"{'rank':>4} {'ms/step':>10} {'%wall':>8} name")
        for i, (name, us) in enumerate(ranked, 1):
            ms = (us / 1000.0) / active
            print(f"{i:4d} {ms:10.3f} {100*ms/avg_wall:7.1f}%  {name[:90]}")

    loader.stop()


if __name__ == "__main__":
    main()
