#!/usr/bin/env python3
"""Isolated hot-kernel bench + ncu target harness.

Each op runs ONE production-shape op in a tight loop with warmup, then
CUDA-event timing with hard sync. ncu invocation per op:

    CUDA_VISIBLE_DEVICES=<g> ncu --clock-control none --cache-control none \
        -k regex:<pattern> -s 3 -c 3 \
        --section SpeedOfLight --section ComputeWorkloadAnalysis \
        --section MemoryWorkloadAnalysis --section Occupancy \
        uv run python scripts/ncu_hot_kernels.py <op>

Shapes come from the profiled production step (bs=32, 67 tokens, d=512,
N=2080 encoder rows = 32 clips x 65 ctx frames, channels_last, bf16).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

# match training: cudnn benchmark autotunes algo choice per shape
torch.backends.cudnn.benchmark = True

D = torch.device("cuda")
CL = torch.channels_last

N_CTX = 2144  # 32 * 67 temporal rows
N_ENC = 2080  # 32 * 65 encoder rows (ring encodes ctx only)


def _gemm(m: int, k: int, n: int):
    x = torch.randn(m, k, device=D, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(n, k, device=D, dtype=torch.bfloat16, requires_grad=True)
    out = lambda: F.linear(x, w)  # noqa: E731
    flops = 3 * 2 * m * k * n  # fwd + bwd (dgrad + wgrad)
    return out, flops


def _conv(cin: int, cout: int, k: int, s: int, hw_in: int, groups: int):
    x = torch.randn(N_ENC, cin, hw_in, hw_in, device=D, dtype=torch.bfloat16)
    x = x.to(memory_format=CL).requires_grad_(True)
    w = torch.randn(cout, cin // groups, k, k, device=D, dtype=torch.bfloat16)
    w = w.to(memory_format=CL).requires_grad_(True)
    hw_out = (hw_in + 2 * (k // 2) - k) // s + 1
    out = lambda: F.conv2d(x, w, None, s, k // 2, 1, groups)  # noqa: E731
    flops = 3 * 2 * N_ENC * cout * hw_out * hw_out * (cin // groups) * k * k
    return out, flops


def _gelu(c: int, hw: int):
    x = torch.randn(N_ENC, c, hw, hw, device=D, dtype=torch.bfloat16)
    x = x.to(memory_format=CL).requires_grad_(True)
    out = lambda: F.gelu(x, approximate="none")  # noqa: E731
    flops = float("nan")  # memory-bound; report GB/s separately
    return out, flops


def _gn_gelu(c: int, hw: int):
    from pan2 import kernels

    x = torch.randn(N_ENC, c, hw, hw, device=D, dtype=torch.bfloat16)
    x = x.to(memory_format=CL).requires_grad_(True)
    w = torch.randn(c, device=D, requires_grad=True)
    b = torch.randn(c, device=D, requires_grad=True)
    fn = kernels.get("group_norm_gelu")
    out = lambda: fn(x, w, b, 8)  # noqa: E731
    return out, float("nan")


OPS = {
    "gemm_qkv": lambda: _gemm(N_CTX, 512, 1536),
    "gemm_proj": lambda: _gemm(N_CTX, 512, 512),
    "gemm_fc1": lambda: _gemm(N_CTX, 512, 2048),
    "gemm_fc2": lambda: _gemm(N_CTX, 2048, 512),
    "conv_stem": lambda: _conv(3, 32, 7, 2, 64, 1),
    "conv_b1": lambda: _conv(32, 64, 3, 2, 32, 1),
    "conv_b2dw": lambda: _conv(64, 64, 3, 2, 16, 64),
    "conv_b2pw": lambda: _conv(64, 128, 1, 1, 8, 1),
    "conv_b3dw": lambda: _conv(128, 128, 3, 2, 8, 128),
    "conv_b3pw": lambda: _conv(128, 512, 1, 1, 4, 1),
    "gelu_stem": lambda: _gelu(32, 32),
    "gelu_b1": lambda: _gelu(64, 16),
    "gn_gelu_b2": lambda: _gn_gelu(128, 8),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=sorted(OPS) + ["all"])
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--bwd", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    names = sorted(OPS) if args.op == "all" else [args.op]
    for name in names:
        out, flops = OPS[name]()
        # warmup (also settles cublas algo heuristics)
        for _ in range(args.warmup):
            y = out()
            if args.bwd:
                g = torch.empty_like(y)
                y.backward(g)
        torch.cuda.synchronize()

        times = []
        g = None
        for _ in range(args.iters):
            y = out()
            if args.bwd:
                g = torch.empty_like(y) if g is None else g
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            if args.bwd:
                y.backward(g)
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1))
        times.sort()
        mean = sum(times) / len(times)
        line = f"{name:12s} mean={mean:8.4f} ms  p50={times[len(times)//2]:8.4f}"
        if flops == flops:  # not nan
            tflops = flops / (mean * 1e-3) / 1e12
            line += f"  {tflops:7.1f} TFLOP/s (fwd+bwd)"
        else:
            nb = y.numel() * y.element_size() * 3 / (mean * 1e-3) / 1e9
            line += f"  ~{nb:7.0f} GB/s est"
        print(line, flush=True)


if __name__ == "__main__":
    main()
