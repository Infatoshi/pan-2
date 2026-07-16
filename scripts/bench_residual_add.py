#!/usr/bin/env python3
"""Bench residual_add optimized vs reference at production residual shapes.

Production: B=32, T=67, D=512 bf16 on CUDA.
"""

from __future__ import annotations

import argparse
import time

import torch

from pan2.kernels import get, reference


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=67)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--warm", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}")
    shape = (args.batch, args.seq, args.dim)
    print(f"shape: {shape} dtype=bf16")

    ref = reference("residual_add")
    opt = get("residual_add")

    def bench(fn) -> float:
        x = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        y = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        for _ in range(args.warm):
            fn(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn(x, y)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.iters * 1000.0

    ref_ms = bench(ref)
    opt_ms = bench(opt)
    print(f"fwd  ref={ref_ms:.4f} ms  opt={opt_ms:.4f} ms  speedup={ref_ms/opt_ms:.2f}x")


if __name__ == "__main__":
    main()
