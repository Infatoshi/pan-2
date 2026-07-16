#!/usr/bin/env python3
"""A/B microbench + profiler buckets for temporal transformer fusion.

Production shape: B=32, T=67, d=512, L=8, H=8, bf16 autocast, CUDA.

Baseline loads the pre-fusion `TransformerTemporal` from a pinned git revision
(`BASELINE_REV`, pre-fusion temporal.py). Fused path is the current package
import (Triton bias_gelu / residual_add + torch.compile on build_temporal).

Reports wall ms/step and elementwise+activation+layernorm+copy CUDA kernel bucket.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

# Pre-fusion temporal.py (branch tip before transformer-fusion kernels landed).
BASELINE_REV = "e57eda2"


def _kernel_buckets(prof, n_iters: int) -> dict[str, float]:
    buckets: dict[str, float] = defaultdict(float)
    for e in prof.key_averages():
        t = e.self_device_time_total / 1000.0  # us -> ms total
        if t <= 0:
            continue
        name = e.key
        low = name.lower()
        is_kernel = (
            name.startswith("void ")
            or "ampere_" in low
            or "cutlass" in low
            or "triton" in low
            or "pytorch_flash" in low
            or "vectorized_elementwise" in low
            or "elementwise_kernel" in low
            or "layer_norm" in low
            or "vectorized_" in low
            or "reduce_kernel" in low
        )
        is_aten = (
            name.startswith("aten::")
            or name.startswith("autograd::")
            or "Backward" in name
        )
        if is_aten and not is_kernel:
            continue
        if (
            any(k in low for k in ("gemm", "cutlass", "cublas", "ampere_bf16", "s168", "mm"))
            and "elementwise" not in low
        ):
            buckets["gemm"] += t
        elif "flash" in low or "attention" in low or "fmha" in low:
            buckets["attention"] += t
        elif "gelu" in low or "tanh" in low:
            buckets["activation"] += t
        elif "layer_norm" in low or "layernorm" in low:
            buckets["layernorm"] += t
        elif any(k in low for k in ("copy", "clone", "bf16_copy", "direct_copy")):
            buckets["copy"] += t
        elif "elementwise" in low or "add<" in low or "functor_add" in low:
            buckets["elementwise"] += t
        else:
            buckets["other"] += t
    return {k: v / n_iters for k, v in buckets.items()}


def _load_baseline_temporal_cls():
    """Load pre-fusion TransformerTemporal from pinned git revision."""
    root = Path(__file__).resolve().parents[1]
    src = subprocess.check_output(
        ["git", "show", f"{BASELINE_REV}:src/pan2/models/temporal.py"],
        cwd=root,
    )
    with tempfile.NamedTemporaryFile("wb", suffix="_temporal_baseline.py", delete=False) as f:
        f.write(src)
        path = f.name
    spec = importlib.util.spec_from_file_location("temporal_baseline_bench", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TransformerTemporal


def _run(label: str, model, device, n_warm: int, n_run: int, n_prof: int) -> dict:
    bsz, seq, dim = 32, 67, 512
    model = model.to(device)
    model.train()
    x = torch.randn(bsz, seq, dim, device=device)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x)
            loss = y.float().pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(n_warm):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_run):
        step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n_run * 1000.0

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(n_prof):
            step()
        torch.cuda.synchronize()
    buckets = _kernel_buckets(prof, n_prof)
    elem = (
        buckets.get("activation", 0.0)
        + buckets.get("layernorm", 0.0)
        + buckets.get("copy", 0.0)
        + buckets.get("elementwise", 0.0)
    )
    print(f"\n=== {label} ===")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"wall_ms/step: {wall:.3f}")
    print("kernel buckets (ms/step):")
    for k in ("gemm", "attention", "activation", "layernorm", "copy", "elementwise", "other"):
        if k in buckets:
            print(f"  {k:12s} {buckets[k]:7.3f}")
    print(f"  {'ELEM_COPY':12s} {elem:7.3f}")
    return {"wall_ms": wall, "elem_copy_ms": elem, "buckets": buckets}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--run", type=int, default=40)
    p.add_argument("--prof", type=int, default=12)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")

    BaselineCls = _load_baseline_temporal_cls()
    base_model = BaselineCls(
        d_model=512, n_layers=8, n_heads=8, dropout=0.0, max_len=128
    )
    base = _run(
        f"baseline (git {BASELINE_REV} temporal, eager)",
        base_model,
        device,
        args.warm,
        args.run,
        args.prof,
    )

    from pan2.models.temporal import build_temporal

    fused_model = build_temporal(
        "transformer", d_model=512, n_layers=8, n_heads=8, dropout=0.0, max_len=128
    )
    fused = _run(
        "fused (kernels + compile)",
        fused_model,
        device,
        max(args.warm, 25),
        args.run,
        args.prof,
    )

    wall_d = (fused["wall_ms"] - base["wall_ms"]) / base["wall_ms"] * 100.0
    elem_d = (fused["elem_copy_ms"] - base["elem_copy_ms"]) / base["elem_copy_ms"] * 100.0
    print("\n=== delta fused vs baseline ===")
    print(f"wall:      {base['wall_ms']:.3f} -> {fused['wall_ms']:.3f} ms  ({wall_d:+.1f}%)")
    print(
        f"elem_copy: {base['elem_copy_ms']:.3f} -> {fused['elem_copy_ms']:.3f} ms  ({elem_d:+.1f}%)"
    )
    if elem_d > -40.0:
        raise SystemExit(
            f"FAIL: elem_copy reduction {elem_d:.1f}% does not meet -40% target"
        )
    print("PASS: elem_copy bucket reduced by >= 40%")


if __name__ == "__main__":
    main()
