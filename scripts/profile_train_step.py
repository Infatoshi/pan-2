#!/usr/bin/env python3
"""Profile pretrain step wall time + top CUDA kernels (no double-count parents)."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from pan2.config import ModelConfig
from pan2.data.synthetic import synthetic_batch
from pan2.models.policy import PanPolicy
from pan2.train.losses import contrastive_loss
from pan2.train.speed import configure_cuda_fast_math

EXCLUDE_EXACT = {
    "forward",
    "backward",
    "loss",
    "optim",
    "train_step",
    "ProfilerStep*",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
}


def is_user_range(name: str) -> bool:
    if name in EXCLUDE_EXACT:
        return True
    if name.startswith("ProfilerStep"):
        return True
    if name.startswith("##"):
        return True
    return False


def normalize_kernel_name(name: str) -> str:
    n = name
    if n.startswith("void "):
        n = n[5:]
    if "xmma_fprop" in n or ("implicit_gemm" in n and "fprop" in n):
        return "cudnn_conv_fprop_xmma"
    if "xmma_wgrad" in n or ("implicit_gemm" in n and "wgrad" in n):
        return "cudnn_conv_wgrad_xmma"
    if "xmma_dgrad" in n or ("implicit_gemm" in n and "dgrad" in n):
        return "cudnn_conv_dgrad_xmma"
    if "batch_norm_backward" in n:
        return "batch_norm_backward_kernel"
    if "batch_norm_collect_statistics" in n or "batch_norm_transform_input" in n:
        return "batch_norm_fwd_kernel"
    if "group_norm" in n.lower() or "GroupNorm" in n:
        return "group_norm_kernel"
    if "GeluBackward" in n or "gelu_backward" in n:
        return "gelu_backward_kernel"
    if "vectorized_elementwise" in n and "Gelu" in n:
        return "gelu_fwd_kernel"
    if "vectorized_elementwise" in n:
        return "elementwise_kernel"
    if "convolve_common_engine" in n:
        return "cudnn_convolve_engine"
    if "cutlass" in n and "wgrad" in n:
        return "cutlass_conv_wgrad"
    if "cutlass" in n and "dgrad" in n:
        return "cutlass_conv_dgrad"
    if "cutlass" in n and ("fprop" in n or "gemm" in n):
        return "cutlass_gemm_or_fprop"
    if n.startswith("aten::"):
        return n
    if len(n) > 120:
        n = n[:117] + "..."
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--frame-subsample", type=int, default=8)
    p.add_argument("--stem-channels", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--active", type=int, default=25)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--real-data", action="store_true")
    p.add_argument("--data", default="/data/pan-2/episodes")
    p.add_argument("--on-device-only", action="store_true")
    args = p.parse_args()

    configure_cuda_fast_math()
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"config bs={args.batch_size} T={args.context_len} img={args.image_size} "
        f"layers={args.n_layers} d={args.d_model} subsample={args.frame_subsample} "
        f"stem={args.stem_channels} compile={args.compile} real_data={args.real_data}"
    )

    mcfg = ModelConfig(
        image_size=args.image_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        context_len=args.context_len,
        action_chunk=10,
        n_discrete=23,
        mouse_dim=2,
        backbone="transformer",
        frame_subsample=args.frame_subsample,
        stem_channels=args.stem_channels,
    )
    model: torch.nn.Module = PanPolicy(mcfg).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model_params_M={nparams:.2f}")
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    model.train()

    if args.on_device_only:
        fixed = synthetic_batch(
            args.batch_size,
            args.context_len,
            args.image_size,
            10,
            23,
            device=device,
            uint8=True,
        )

        def next_batch():
            return fixed

    elif args.real_data:
        from torch.utils.data import DataLoader

        from pan2.data.vpt_episodes import VPTEpisodeDataset

        ds = VPTEpisodeDataset(
            args.data,
            context_len=args.context_len,
            action_chunk=10,
            image_size=args.image_size,
            keep_uint8=True,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        it = iter(loader)

        def next_batch():
            nonlocal it
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            return {k: v.to(device, non_blocking=True) for k, v in b.items()}
    else:

        def next_batch():
            b = synthetic_batch(
                args.batch_size,
                args.context_len,
                args.image_size,
                10,
                23,
                device="cpu",
                uint8=True,
            )
            return {k: v.to(device, non_blocking=True) for k, v in b.items()}

    def train_step() -> torch.Tensor:
        with record_function("train_step"):
            batch = next_batch()
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with record_function("forward"):
                    out = model(batch["frames"], batch["goal"], return_actions=False)
                with record_function("loss"):
                    loss = contrastive_loss(out["contrastive_logits"])
            with record_function("backward"):
                loss.backward()
            with record_function("optim"):
                optim.step()
        return loss

    for _ in range(max(8, args.warmup)):
        train_step()
    torch.cuda.synchronize()

    wait, warmup, active = 1, args.warmup, args.active
    sched = schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    total_steps = wait + warmup + active
    wall_ms: list[float] = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for i in range(total_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            train_step()
            end.record()
            torch.cuda.synchronize()
            if i >= wait + warmup:
                wall_ms.append(start.elapsed_time(end))
            prof.step()

    avg_wall = sum(wall_ms) / len(wall_ms)
    wall_ms_sorted = sorted(wall_ms)
    print("\n=== WALL CLOCK (full train step, CUDA events) ===")
    print(
        f"n={len(wall_ms)}  avg_ms={avg_wall:.3f}  "
        f"p50={wall_ms_sorted[len(wall_ms_sorted)//2]:.3f}  "
        f"min={min(wall_ms):.3f}  max={max(wall_ms):.3f}"
    )

    kernel_total_us: dict[str, float] = defaultdict(float)
    kernel_counts: dict[str, int] = defaultdict(int)

    for evt in prof.key_averages():
        name = evt.key
        if is_user_range(name):
            continue
        self_dev = getattr(evt, "self_device_time_total", None)
        if self_dev is None:
            self_dev = getattr(evt, "self_cuda_time_total", 0) or 0
        self_dev = float(self_dev or 0)
        if self_dev <= 0:
            continue
        bucket = normalize_kernel_name(name)
        kernel_total_us[bucket] += self_dev
        kernel_counts[bucket] += int(evt.count)

    print("\n=== torch.profiler table (self CUDA, top 20) ===")
    try:
        print(
            prof.key_averages().table(
                sort_by="self_device_time_total",
                row_limit=20,
                max_name_column_width=80,
            )
        )
    except Exception:
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=20,
                max_name_column_width=80,
            )
        )

    total_us = sum(kernel_total_us.values())
    n_active = active
    ranked = sorted(kernel_total_us.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [(n, u) for n, u in ranked if not is_user_range(n)]

    print("\n=== TOP KERNELS (self device time, deduped buckets) ===")
    print(
        f"sum_self_device_ms_over_active={total_us/1000:.2f}  "
        f"avg_per_step={total_us/1000/n_active:.2f}  wall_avg={avg_wall:.2f}"
    )
    print(
        f"{'rank':>4}  {'ms/step':>10}  {'%_cuda_self':>11}  {'%_wall':>8}  "
        f"{'calls/step':>10}  name"
    )

    top5 = []
    for i, (name, us) in enumerate(ranked[: args.top], 1):
        ms = (us / 1000.0) / n_active
        pct_cuda = 100.0 * us / total_us if total_us else 0.0
        pct_wall = 100.0 * ms / avg_wall if avg_wall else 0.0
        calls = kernel_counts[name] / n_active
        print(
            f"{i:4d}  {ms:10.3f}  {pct_cuda:10.1f}%  {pct_wall:7.1f}%  "
            f"{calls:10.1f}  {name}"
        )
        if i <= 5:
            top5.append((name, ms, pct_cuda, pct_wall, calls))

    print("\n=== PHASE RANGES (inclusive device time / step) ===")
    for label in ("train_step", "forward", "loss", "backward", "optim"):
        matches = [e for e in prof.key_averages() if e.key == label]
        if not matches:
            continue
        e = matches[0]
        cpu_ms = (getattr(e, "cpu_time_total", 0) or 0) / 1000.0 / n_active
        dev_total = (
            getattr(e, "device_time_total", None)
            or getattr(e, "cuda_time_total", 0)
            or 0
        )
        dev_ms = float(dev_total) / 1000.0 / n_active
        print(f"  {label:10s}  cpu={cpu_ms:7.2f} ms  device_incl={dev_ms:7.2f} ms")

    print("\n======== TOP 5 HOTTEST (answer) ========")
    print(f"Average train-step wall clock: {avg_wall:.2f} ms")
    for i, (name, ms, pct_cuda, pct_wall, calls) in enumerate(top5, 1):
        print(
            f"{i}. {name}\n"
            f"   {ms:.2f} ms/step  |  {pct_wall:.1f}% of wall  |  "
            f"{pct_cuda:.1f}% of CUDA self-time  |  ~{calls:.1f} launches/step"
        )
    s = sum(t[3] for t in top5)
    print(
        f"\nNote: top-5 %_wall sum={s:.1f}% "
        f"(residual = other kernels + CPU + idle gaps)"
    )

    out = Path("data/cache/train_step_profile.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out))
    print(f"chrome_trace={out.resolve()}")


if __name__ == "__main__":
    main()
