#!/usr/bin/env python3
"""Deep profile of GroupNorm + Conv backwards (per-module + shape breakdown)."""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, schedule

from pan2.config import ModelConfig
from pan2.data.gpu_pipeline import PipelineConfig, PipelinedGpuPretrainLoader
from pan2.models.policy import PanPolicy
from pan2.train.losses import contrastive_loss
from pan2.train.speed import configure_cuda_fast_math


class CudaModuleTimer:
    """CUDA-event timing for selected modules (forward + full backward)."""

    def __init__(self, model: nn.Module):
        self.fwd_ms: dict[str, float] = defaultdict(float)
        self.bwd_ms: dict[str, float] = defaultdict(float)
        self.fwd_calls: dict[str, int] = defaultdict(int)
        self.bwd_calls: dict[str, int] = defaultdict(int)
        self.enabled = False
        self._fwd_pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._bwd_pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._handles = []

        for name, mod in model.named_modules():
            if not isinstance(mod, (nn.Conv2d, nn.GroupNorm)):
                continue

            def pre_fwd(_m, _inp, n=name):
                if not self.enabled:
                    return
                t0 = torch.cuda.Event(enable_timing=True)
                t0.record()
                _m._prof_fwd_t0 = t0  # type: ignore[attr-defined]

            def post_fwd(_m, _inp, _out, n=name):
                if not self.enabled or not hasattr(_m, "_prof_fwd_t0"):
                    return
                t1 = torch.cuda.Event(enable_timing=True)
                t1.record()
                self._fwd_pending.append((n, _m._prof_fwd_t0, t1))  # type: ignore[attr-defined]

            def pre_bwd(_m, _g_out, n=name):
                if not self.enabled:
                    return
                t0 = torch.cuda.Event(enable_timing=True)
                t0.record()
                _m._prof_bwd_t0 = t0  # type: ignore[attr-defined]

            def post_bwd(_m, _g_in, _g_out, n=name):
                if not self.enabled or not hasattr(_m, "_prof_bwd_t0"):
                    return
                t1 = torch.cuda.Event(enable_timing=True)
                t1.record()
                self._bwd_pending.append((n, _m._prof_bwd_t0, t1))  # type: ignore[attr-defined]

            self._handles.append(mod.register_forward_pre_hook(pre_fwd))
            self._handles.append(mod.register_forward_hook(post_fwd))
            self._handles.append(mod.register_full_backward_pre_hook(pre_bwd))
            self._handles.append(mod.register_full_backward_hook(post_bwd))

    def begin_step(self) -> None:
        self.enabled = True
        self._fwd_pending = []
        self._bwd_pending = []

    def end_step(self) -> None:
        self.enabled = False
        torch.cuda.synchronize()
        for n, t0, t1 in self._fwd_pending:
            self.fwd_ms[n] += t0.elapsed_time(t1)
            self.fwd_calls[n] += 1
        for n, t0, t1 in self._bwd_pending:
            self.bwd_ms[n] += t0.elapsed_time(t1)
            self.bwd_calls[n] += 1

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def _kind(name: str) -> str:
    n = name.lower()
    gn_pat = "groupnorm" in n or n.endswith(".gn") or ".gn" in n or n.endswith("_gn")
    if isinstance(name, str) and gn_pat:
        # GroupNorm modules are named *.gn or *.b*_gn
        pass
    # decide by leaf name
    leaf = name.split(".")[-1]
    if leaf == "gn" or leaf.endswith("_gn"):
        return "gn"
    if leaf == "conv" or leaf.endswith("_dw") or leaf.endswith("_pw"):
        return "conv"
    if "groupnorm" in n:
        return "gn"
    if "conv" in n:
        return "conv"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--budget-gb", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    configure_cuda_fast_math()
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"B={args.batch_size}")

    pcfg = PipelineConfig(
        batch_size=32,
        context_len=128,
        frame_subsample=2,
        image_size=64,
        budget_gb=args.budget_gb,
        num_producers=8,
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
        frame_subsample=1,
        stem_channels=32,
        n_discrete=23,
    )
    model = PanPolicy(mcfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    model.train()
    print(f"model_params_M={sum(p.numel() for p in model.parameters())/1e6:.2f}")

    # list tracked modules
    tracked = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.GroupNorm))]
    print(f"tracked_modules={len(tracked)}")
    for n in tracked:
        m = dict(model.named_modules())[n]
        if isinstance(m, nn.Conv2d):
            print(f"  CONV {n}: in={m.in_channels} out={m.out_channels} k={m.kernel_size} "
                  f"s={m.stride} g={m.groups}")
        else:
            print(f"  GN   {n}: C={m.num_channels} groups={m.num_groups}")

    timer = CudaModuleTimer(model)
    ring, rng = loader.ring, loader.rng
    B = args.batch_size

    def get_batch():
        for _ in range(20000):
            slots = ring.sample_slots(B, rng)
            if slots:
                break
            time.sleep(0.001)
        slot_t = torch.tensor(slots, device=device, dtype=torch.long)
        frames = ring.frames.index_select(0, slot_t)
        gi = torch.randint(0, ring.t_sub, (B,), device=device)
        bi = torch.arange(B, device=device)
        return {"frames": frames, "goal": frames[bi, gi]}

    def train_step(use_timer: bool = False) -> float:
        batch = get_batch()
        optim.zero_grad(set_to_none=True)
        if use_timer:
            timer.begin_step()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch["frames"], batch["goal"], return_actions=False)
            loss = contrastive_loss(out["contrastive_logits"])
        loss.backward()
        optim.step()
        if use_timer:
            timer.end_step()
        return float(loss.detach())

    for _ in range(args.warmup):
        train_step(False)
    torch.cuda.synchronize()

    walls: list[float] = []
    for _ in range(args.steps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record()
        train_step(True)
        e.record()
        torch.cuda.synchronize()
        walls.append(s.elapsed_time(e))
    avg_wall = sum(walls) / len(walls)
    n = args.steps
    print(f"\n=== WALL avg_ms={avg_wall:.2f} p50={sorted(walls)[n//2]:.2f} ===")

    print("\n=== PER-MODULE FORWARD (ms/step) ===")
    for name, total in sorted(timer.fwd_ms.items(), key=lambda kv: -kv[1]):
        ms = total / n
        print(f"  {ms:7.2f}  {100*ms/avg_wall:5.1f}%  {_kind(name):4s}  {name}")

    print("\n=== PER-MODULE BACKWARD (ms/step) ===")
    for name, total in sorted(timer.bwd_ms.items(), key=lambda kv: -kv[1]):
        ms = total / n
        print(f"  {ms:7.2f}  {100*ms/avg_wall:5.1f}%  {_kind(name):4s}  {name}")

    conv_fwd = sum(v for k, v in timer.fwd_ms.items() if _kind(k) == "conv") / n
    gn_fwd = sum(v for k, v in timer.fwd_ms.items() if _kind(k) == "gn") / n
    conv_bwd = sum(v for k, v in timer.bwd_ms.items() if _kind(k) == "conv") / n
    gn_bwd = sum(v for k, v in timer.bwd_ms.items() if _kind(k) == "gn") / n

    print("\n=== AGGREGATE vs WALL ===")
    print(f"  all_conv_fwd   {conv_fwd:7.2f} ms  {100*conv_fwd/avg_wall:5.1f}%")
    print(f"  all_gn_fwd     {gn_fwd:7.2f} ms  {100*gn_fwd/avg_wall:5.1f}%")
    print(f"  all_conv_bwd   {conv_bwd:7.2f} ms  {100*conv_bwd/avg_wall:5.1f}%")
    print(f"  all_gn_bwd     {gn_bwd:7.2f} ms  {100*gn_bwd/avg_wall:5.1f}%")
    print(f"  conv_bwd+gn_bwd {conv_bwd+gn_bwd:6.2f} ms  {100*(conv_bwd+gn_bwd)/avg_wall:5.1f}%")
    rx = conv_bwd / max(conv_fwd, 1e-6)
    rg = gn_bwd / max(gn_fwd, 1e-6)
    print(f"  ratio bwd/fwd conv={rx:.2f}x  gn={rg:.2f}x")

    # Stage rollups
    print("\n=== STAGE BWD ROLLUP ===")
    stages = {
        "stem": lambda k: "encoder.stem" in k,
        "block1": lambda k: "encoder.block1" in k,
        "block2": lambda k: "encoder.block2" in k,
        "block3": lambda k: "encoder.block3" in k,
    }
    for sname, pred in stages.items():
        c = sum(v for k, v in timer.bwd_ms.items() if pred(k) and _kind(k) == "conv") / n
        g = sum(v for k, v in timer.bwd_ms.items() if pred(k) and _kind(k) == "gn") / n
        tot = c + g
        print(f"  {sname:7s}  conv_bwd={c:6.2f}ms  gn_bwd={g:6.2f}ms  sum={tot:6.2f}ms"
              f"  ({100*tot/avg_wall:.1f}% wall)")

    # Profiler with shapes
    print("\n=== PROFILER self-CUDA by op + input shapes ===")
    wait, warmup, active = 1, 4, 12
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        record_shapes=True,
        with_modules=True,
    ) as prof:
        for _ in range(wait + warmup + active):
            train_step(False)
            prof.step()

    interesting = []
    for evt in prof.key_averages(group_by_input_shape=True):
        key = str(evt.key)
        kl = key.lower()
        if not any(
            s in kl
            for s in (
                "convolution_backward",
                "cudnn_convolution",
                "group_norm",
                "wgrad",
                "dgrad",
                "conv_depthwise",
            )
        ):
            continue
        us = (
            getattr(evt, "self_device_time_total", None)
            or getattr(evt, "self_cuda_time_total", 0)
            or 0
        )
        if not us:
            continue
        ms = float(us) / 1000.0 / active
        shapes = getattr(evt, "input_shapes", None) or []
        interesting.append((ms, evt.count / active, key, shapes))
    interesting.sort(reverse=True)
    print(f"{'ms/step':>8} {'calls':>7}  name / shapes")
    for ms, calls, key, shapes in interesting[:30]:
        print(f"{ms:8.2f} {calls:7.1f}  {key[:75]}")
        if shapes:
            print(f"{'':16}shapes={shapes}")

    print("\n=== TOP 10 overall self-CUDA ===")
    try:
        print(
            prof.key_averages().table(
                sort_by="self_device_time_total", row_limit=12, max_name_column_width=72
            )
        )
    except Exception:
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=12, max_name_column_width=72
            )
        )

    timer.close()
    loader.stop()
    print("done")


if __name__ == "__main__":
    main()
