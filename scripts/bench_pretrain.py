#!/usr/bin/env python3
"""Stage-wise pretrain throughput benchmark with CPU / transfer / GPU shares."""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from pan2.config import ModelConfig
from pan2.data.shards import MANIFEST_NAME, ShardDataset
from pan2.data.synthetic import synthetic_batch
from pan2.data.vpt_episodes import VPTEpisodeDataset
from pan2.models.policy import PanPolicy
from pan2.train.losses import contrastive_loss
from pan2.train.speed import configure_cuda_fast_math


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class IterBox:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/data/pan-2/episodes")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--float32-host", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    configure_cuda_fast_math()
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(f"device={device} name={name}")
    keep_uint8 = not args.float32_host

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
    )

    data_root = Path(args.data)
    cls = ShardDataset if (data_root / MANIFEST_NAME).exists() else VPTEpisodeDataset
    ds = cls(
        args.data,
        context_len=args.context_len,
        action_chunk=10,
        image_size=args.image_size,
        keep_uint8=keep_uint8,
    )
    n_eps = len(ds.pairs) if hasattr(ds, "pairs") else len(ds.segments)
    print(
        f"dataset={cls.__name__} episodes={n_eps} T={args.context_len} "
        f"img={args.image_size} bs={args.batch_size} keep_uint8={keep_uint8} "
        f"compile={args.compile}"
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_cuda,
        drop_last=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
    )
    box = IterBox(loader)

    model: torch.nn.Module = PanPolicy(mcfg).to(device)
    if args.compile and use_cuda:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        fused=use_cuda,
    )
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model_params_M={nparams:.2f}")

    sample = box.next()
    frames = sample["frames"]
    bytes_per_batch = frames.nelement() * frames.element_size()
    print(
        f"frames_dtype={frames.dtype} frames_shape={tuple(frames.shape)} "
        f"frames_MB={bytes_per_batch / 1e6:.2f}"
    )

    def one_step_train(batch_on_device: dict) -> torch.Tensor:
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            out = model(batch_on_device["frames"], batch_on_device["goal"], return_actions=False)
            loss = contrastive_loss(out["contrastive_logits"])
        loss.backward()
        optim.step()
        return loss

    # warmup (longer if compile)
    warm_n = 20 if args.compile else 8
    for _ in range(warm_n):
        batch = box.next()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        sync()
        one_step_train(batch)
        sync()

    syn = synthetic_batch(
        args.batch_size,
        args.context_len,
        args.image_size,
        10,
        23,
        device=device,
        uint8=keep_uint8,
    )
    for _ in range(5):
        one_step_train(syn)
    sync()
    t0 = time.perf_counter()
    n_gpu = args.steps
    for _ in range(n_gpu):
        one_step_train(syn)
    sync()
    gpu_only_ms = 1000 * (time.perf_counter() - t0) / n_gpu
    print(f"[gpu_compute_only] {gpu_only_ms:.2f} ms/step  steps/s={1000/gpu_only_ms:.2f}")

    cpu_batch = {k: v.cpu() for k, v in box.next().items()}
    for _ in range(5):
        _ = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
        sync()
    sync()
    t0 = time.perf_counter()
    n_h = 40
    for _ in range(n_h):
        _ = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
        sync()
    h2d_ms = 1000 * (time.perf_counter() - t0) / n_h
    print(f"[h2d_only] {h2d_ms:.2f} ms/batch  MB={bytes_per_batch/1e6:.2f}")

    for _ in range(5):
        box.next()
    t0 = time.perf_counter()
    n_d = 40
    for _ in range(n_d):
        box.next()
    data_only_ms = 1000 * (time.perf_counter() - t0) / n_d
    print(f"[dataloader_only] {data_only_ms:.2f} ms/batch  batch/s={1000/data_only_ms:.2f}")

    t_cpu = t_h2d = t_gpu = 0.0
    n = args.steps
    last_loss = None
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        batch = box.next()
        _ = batch["frames"].shape
        t1 = time.perf_counter()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        sync()
        t2 = time.perf_counter()
        loss = one_step_train(batch)
        sync()
        t3 = time.perf_counter()
        t_cpu += t1 - t0
        t_h2d += t2 - t1
        t_gpu += t3 - t2
        last_loss = float(loss.detach())

    total = t_cpu + t_h2d + t_gpu

    def pct(x: float) -> float:
        return 100.0 * x / total if total else 0.0

    print(f"[full_pipeline_phased] n={n} last_loss={last_loss:.4f}")
    print(f"  CPU_data     {1000*t_cpu/n:7.2f} ms/step  ({pct(t_cpu):5.1f}%)")
    print(f"  transfer_H2D {1000*t_h2d/n:7.2f} ms/step  ({pct(t_h2d):5.1f}%)")
    print(f"  GPU_kernels  {1000*t_gpu/n:7.2f} ms/step  ({pct(t_gpu):5.1f}%)")
    print(
        f"  TOTAL        {1000*total/n:7.2f} ms/step  steps/s={n/total:.2f}  "
        f"frames/s={n/total*args.batch_size*args.context_len:.0f}"
    )

    def full_step() -> None:
        batch = box.next()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        one_step_train(batch)

    for _ in range(5):
        full_step()
    sync()
    ts = []
    for _ in range(args.steps):
        sync()
        t0 = time.perf_counter()
        full_step()
        sync()
        ts.append(time.perf_counter() - t0)
    wall_ms = 1000 * statistics.mean(ts)
    print(
        f"[full_wall_prefetch] {wall_ms:.2f} ms/step  steps/s={1000/wall_ms:.2f}  "
        f"frames/s={1000/wall_ms*args.batch_size*args.context_len:.0f}"
    )

    stall_ms = max(0.0, wall_ms - gpu_only_ms)
    host_bound_pct = 100.0 * stall_ms / wall_ms if wall_ms else 0.0
    gpu_wall_pct = 100.0 * min(gpu_only_ms, wall_ms) / wall_ms if wall_ms else 0.0
    print("--- SUMMARY ---")
    print(
        f"SERIAL_SHARE  cpu={pct(t_cpu):.1f}%  transfer={pct(t_h2d):.1f}%  gpu={pct(t_gpu):.1f}%"
    )
    print(
        f"WALL_SHARE    gpu_kernels~={gpu_wall_pct:.1f}%  "
        f"cpu+transfer_stall~={host_bound_pct:.1f}%  "
        f"(wall={wall_ms:.1f}ms gpu_only={gpu_only_ms:.1f}ms)"
    )
    print(
        f"ISOLATED      dataloader={data_only_ms:.1f}ms  h2d={h2d_ms:.1f}ms  "
        f"gpu={gpu_only_ms:.1f}ms"
    )


if __name__ == "__main__":
    main()
