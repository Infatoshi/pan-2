#!/usr/bin/env python3
"""Benchmark fused frontend conv+GELU against its PyTorch composition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pan2 import kernels
from pan2.train.speed import configure_cuda_fast_math

# Production shapes, bs256 pack pretrain: N = 256 x (128 ctx + 1 goal + 4
# hard negs) = 34,048 images per encoder batch (2026-07-18).
SHAPES = {
    "stem": ((34048, 3, 64, 64), (32, 3, 7, 7), 3),
    "b1": ((34048, 32, 32, 32), (64, 32, 3, 3), 1),
}


def _measure(fn, x, weight, grad, warmup: int, steps: int) -> float:
    def step() -> None:
        x.grad = None
        weight.grad = None
        fn(x, weight).backward(grad)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    configure_cuda_fast_math()
    torch.manual_seed(0)
    print(f"device={torch.cuda.get_device_name(0)}")
    for name, (input_shape, weight_shape, padding) in SHAPES.items():
        x = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
        x = x.contiguous(memory_format=torch.channels_last).requires_grad_()
        weight = torch.randn(weight_shape, device="cuda", dtype=torch.bfloat16)
        weight = weight.contiguous(memory_format=torch.channels_last).requires_grad_()
        output_shape = (
            input_shape[0],
            weight_shape[0],
            input_shape[2] // 2,
            input_shape[3] // 2,
        )
        grad = torch.randn(output_shape, device="cuda", dtype=torch.bfloat16)
        grad = grad.contiguous(memory_format=torch.channels_last)
        fused = lambda a, b: kernels.get("conv_gelu")(a, b, 2, padding)  # noqa: E731
        reference = lambda a, b: kernels.reference("conv_gelu")(  # noqa: E731
            a, b, 2, padding
        )
        fused_ms = _measure(fused, x, weight, grad, args.warmup, args.steps)
        ref_ms = _measure(reference, x, weight, grad, args.warmup, args.steps)
        print(
            f"{name:4s} shape={input_shape} fused_fwd_bwd_ms={fused_ms:.4f} "
            f"ref_fwd_bwd_ms={ref_ms:.4f} speedup={ref_ms / fused_ms:.3f}x"
        )


if __name__ == "__main__":
    main()
