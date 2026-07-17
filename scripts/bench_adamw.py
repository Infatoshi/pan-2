#!/usr/bin/env python3
"""Bench AdamW.step() wall on the production PanPolicy param set.

Compares torch.optim.AdamW(fused=True) vs pan2 FusedAdamW (when available).
Grads are pre-populated; ~20 warmup + 200 timed iters with per-iter sync.

  CUDA_VISIBLE_DEVICES=1 uv run python scripts/bench_adamw.py
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/bench_adamw.py --impl torch
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/bench_adamw.py --impl ours
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from pan2.config import ModelConfig
from pan2.models.policy import PanPolicy


def _production_model(device: torch.device) -> PanPolicy:
    cfg = ModelConfig(
        image_size=64,
        d_model=512,
        n_layers=8,
        n_heads=8,
        context_len=128,
        action_chunk=10,
        n_discrete=23,
        mouse_dim=2,
        backbone="transformer",
        frame_subsample=1,
    )
    return PanPolicy(cfg).to(device)


def _fill_grads(params: list[torch.nn.Parameter], seed: int) -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for p in params:
        # match param memory layout (channels_last conv weights stay dense NHWC)
        noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
        p.grad = noise.to(device=p.device, dtype=p.dtype).to(memory_format=torch.contiguous_format)
        if p.is_contiguous(memory_format=torch.channels_last):
            p.grad = p.grad.to(memory_format=torch.channels_last)
        elif not p.is_contiguous():
            # dense non-standard: copy into matching strides
            g_match = torch.empty_like(p)
            g_match.copy_(noise.to(device=p.device, dtype=p.dtype))
            p.grad = g_match


def _make_optim(impl: str, params, lr: float, weight_decay: float):
    if impl == "torch":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
    if impl == "foreach":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, fused=False, foreach=True
        )
    if impl == "ours":
        from pan2.kernels.fused_adamw import FusedAdamW

        return FusedAdamW(params, lr=lr, weight_decay=weight_decay)
    if impl == "ours-clip":
        from pan2.kernels.fused_adamw import FusedAdamW

        return FusedAdamW(
            params, lr=lr, weight_decay=weight_decay, clip_max_norm=1.0
        )
    if impl == "torch-clip":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)

        class _ClipWrap:
            def __init__(self, opt, params):
                self.opt, self.params = opt, params

            def step(self):
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.opt.step()

            def zero_grad(self, **kw):
                self.opt.zero_grad(**kw)

        return _ClipWrap(opt, params)
    raise ValueError(impl)


def _bench_step(optim, params, n_warm: int, n_iter: int, seed0: int) -> list[float]:
    # Pre-populate grads once (optimizer step wall only; no grad fill in timer).
    _fill_grads(params, seed0)
    optim.step()
    for _ in range(n_warm):
        optim.step()
    if params[0].is_cuda:
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_iter):
        if params[0].is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optim.step()
        if params[0].is_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--impl",
        choices=("torch", "foreach", "ours", "torch-clip", "ours-clip", "both", "all", "clip"),
        default="all",
    )
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    print(f"device: {name}")

    model = _production_model(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    n_tensors = len(params)
    print(f"params: {n_params / 1e6:.3f}M elements across {n_tensors} tensors")

    if args.impl == "both":
        impls = ["torch", "ours"]
    elif args.impl == "all":
        impls = ["torch", "foreach", "ours"]
    elif args.impl == "clip":
        impls = ["torch-clip", "ours-clip"]
    else:
        impls = [args.impl]
    results: dict[str, list[float]] = {}
    for impl in impls:
        # fresh model copy so state is clean and fair
        model_i = _production_model(device)
        params_i = [p for p in model_i.parameters() if p.requires_grad]
        # copy weights from reference so both start identical if needed
        with torch.no_grad():
            for a, b in zip(params_i, params, strict=True):
                a.copy_(b)
        try:
            optim = _make_optim(impl, params_i, args.lr, args.weight_decay)
        except Exception as e:
            print(f"{impl}: unavailable ({e})")
            continue
        times = _bench_step(optim, params_i, args.warm, args.iters, seed0=0)
        results[impl] = times
        mean = statistics.fmean(times)
        p50 = statistics.median(times)
        print(f"{impl:8s}  mean={mean:.4f} ms  p50={p50:.4f} ms  (n={len(times)})")

    if "ours" in results:
        o_mean = statistics.fmean(results["ours"])
        if "torch" in results:
            t_mean = statistics.fmean(results["torch"])
            print(f"speedup vs torch fused (torch/ours): {t_mean / o_mean:.3f}x")
        if "foreach" in results:
            f_mean = statistics.fmean(results["foreach"])
            print(f"speedup vs torch foreach (foreach/ours): {f_mean / o_mean:.3f}x")


if __name__ == "__main__":
    main()
