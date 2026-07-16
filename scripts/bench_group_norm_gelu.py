#!/usr/bin/env python3
"""Benchmark fused GroupNorm+GELU at the encoder's production shapes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pan2 import kernels


def measure_ms(fn, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print(f"device={torch.cuda.get_device_name(0)}")
    print("dtype=torch.bfloat16 groups=8 stats=torch.float32 layout=channels_last")
    print("shape ref_fwd_ms fused_fwd_ms fwd_speedup ref_fwd_bwd_ms fused_fwd_bwd_ms train_speedup")
    for shape in ((2080, 128, 8, 8), (2080, 512, 4, 4)):
        _, channels, _, _ = shape
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous(
            memory_format=torch.channels_last
        )
        weight = torch.ones(channels, device="cuda", dtype=torch.float32, requires_grad=True)
        bias = torch.zeros(channels, device="cuda", dtype=torch.float32, requires_grad=True)
        grad = torch.randn_like(x)
        reference = kernels.reference("group_norm_gelu")
        fused = kernels.get("group_norm_gelu")

        def ref_fwd() -> None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reference(x, weight, bias, 8, 1e-5)

        def fused_fwd() -> None:
            fused(x, weight, bias, 8, 1e-5)

        def ref_train() -> None:
            weight.grad = None
            bias.grad = None
            x_train = x.detach().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = reference(x_train, weight, bias, 8, 1e-5)
            y.backward(grad)

        def fused_train() -> None:
            weight.grad = None
            bias.grad = None
            x_train = x.detach().requires_grad_()
            y = fused(x_train, weight, bias, 8, 1e-5)
            y.backward(grad)

        ref_fwd_ms = measure_ms(ref_fwd, args.warmup, args.steps)
        fused_fwd_ms = measure_ms(fused_fwd, args.warmup, args.steps)
        ref_train_ms = measure_ms(ref_train, args.warmup, args.steps)
        fused_train_ms = measure_ms(fused_train, args.warmup, args.steps)
        print(
            f"{shape} {ref_fwd_ms:.4f} {fused_fwd_ms:.4f} "
            f"{ref_fwd_ms / fused_fwd_ms:.2f}x {ref_train_ms:.4f} "
            f"{fused_train_ms:.4f} {ref_train_ms / fused_train_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
