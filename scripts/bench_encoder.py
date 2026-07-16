#!/usr/bin/env python3
"""Measure production-shape encoder forward/backward and layout kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from pan2.models.encoder import FrameEncoder
from pan2.models.preprocess import prepare_images
from pan2.train.speed import configure_cuda_fast_math


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--profile-active", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    configure_cuda_fast_math()
    torch.manual_seed(0)
    model = FrameEncoder(d_model=512, image_size=64, stem_channels=32).cuda().train()
    images = torch.randint(0, 256, (2080, 3, 64, 64), dtype=torch.uint8, device="cuda")
    x = prepare_images(images, 64)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"input_shape={tuple(x.shape)} dtype={x.dtype} strides={x.stride()} "
        f"channels_last={x.is_contiguous(memory_format=torch.channels_last)}"
    )
    print(
        f"stem_weight_strides={model.stem.conv.weight.stride()} "
        "channels_last="
        f"{model.stem.conv.weight.is_contiguous(memory_format=torch.channels_last)}"
    )

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)
        output.backward(torch.ones_like(output))

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(args.steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    print(
        f"encoder_fwd_bwd_ms mean={sum(times) / len(times):.4f} "
        f"median={sorted(times)[len(times) // 2]:.4f} min={min(times):.4f} "
        f"max={max(times):.4f} n={len(times)}"
    )

    wait, warmup, active = 1, 2, args.profile_active
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
    ) as prof:
        for _ in range(wait + warmup + active):
            step()
            prof.step()
    torch.cuda.synchronize()
    conversions: list[tuple[str, int, float]] = []
    for event in prof.key_averages():
        name = str(event.key)
        lowered = name.lower()
        if "nchwtonhwc" not in lowered and "nhwctonchw" not in lowered:
            continue
        cuda_us = float(
            getattr(event, "self_device_time_total", 0)
            or getattr(event, "self_cuda_time_total", 0)
            or 0
        )
        conversions.append((name, event.count, cuda_us))
    print(
        f"layout_conversion_kernel_kinds={len(conversions)} "
        f"calls={sum(count for _, count, _ in conversions)} "
        f"self_cuda_ms={sum(cuda_us for _, _, cuda_us in conversions) / 1000:.4f}"
    )
    for name, count, cuda_us in conversions:
        print(f"  {count:5d} {cuda_us / 1000:9.4f} ms {name}")


if __name__ == "__main__":
    main()
