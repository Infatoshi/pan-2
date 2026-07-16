#!/usr/bin/env python3
"""Bench bias_gelu optimized vs reference at production MLP shapes.

Production: B=32, T=67, H=2048 (d_model=512, mlp_ratio=4), bf16 on CUDA device.
Reports ms/iter for fwd and fwd+bwd. Speed claims must cite this output.
"""

from __future__ import annotations

import argparse
import time

import torch

from pan2.kernels import get, reference


def _bench(fn, args, n_warm: int, n_iter: int, backward: bool) -> float:
    for _ in range(n_warm):
        out = fn(*args)
        if backward:
            out.float().square().mean().backward()
            for a in args:
                if isinstance(a, torch.Tensor) and a.requires_grad and a.grad is not None:
                    a.grad = None
    if args[0].is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
        if backward:
            out.float().square().mean().backward()
            for a in args:
                if isinstance(a, torch.Tensor) and a.requires_grad and a.grad is not None:
                    a.grad = None
    if args[0].is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=67)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}")
    shape = (args.batch, args.seq, args.hidden)
    print(f"shape: {shape} dtype=bf16")

    ref = reference("bias_gelu")
    opt = get("bias_gelu")

    def make_args(requires_grad: bool):
        x = torch.randn(*shape, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
        b = torch.randn(shape[-1], device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
        return (x, b)

    for backward, label in ((False, "fwd"), (True, "fwd+bwd")):
        ref_ms = _bench(ref, make_args(backward), args.warm, args.iters, backward)
        opt_ms = _bench(opt, make_args(backward), args.warm, args.iters, backward)
        speedup = ref_ms / opt_ms if opt_ms > 0 else float("inf")
        print(f"{label:8s}  ref={ref_ms:.4f} ms  opt={opt_ms:.4f} ms  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
