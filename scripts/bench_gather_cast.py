#!/usr/bin/env python3
"""Bench gather_cast / scale_cast vs the eager chain they replace.

Production: ring [S,67,3,64,64] uint8, B=32 slots/step -> [2080|64] rows
[3,64,64] bf16 channels_last. The eager baseline is index_select + reshape +
to(fp32, CL) + mul_ + autocast bf16 cast (~600 MB traffic). Speed claims
must cite this output.
"""

from __future__ import annotations

import argparse
import time

import torch

from pan2.kernels import get

SCALE = 1.0 / 255.0


def _bench(fn, warm: int, iters: int) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=int, default=2048)
    p.add_argument("--t-slot", type=int, default=67)
    p.add_argument("--t-ctx", type=int, default=65)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    print(f"device: {name}")

    c, h, w = 3, args.image_size, args.image_size
    ring = torch.randint(
        0, 256, (args.slots, args.t_slot, c, h, w), dtype=torch.uint8, device=device
    )
    g = torch.Generator(device="cpu").manual_seed(0)
    slot_t = torch.randint(0, args.slots, (args.batch,), generator=g).to(device)

    gather_cast = get("gather_cast")
    scale_cast = get("scale_cast")

    def eager_batch():
        frames = ring.index_select(0, slot_t)  # [B,T,C,H,W] uint8
        flat = frames.reshape(-1, c, h, w)
        y = flat.to(torch.float32, memory_format=torch.channels_last).mul_(SCALE)
        return y.to(torch.bfloat16)  # autocast conv-input cast

    def fused_batch():
        return gather_cast(ring, slot_t, SCALE, torch.bfloat16, args.t_ctx)

    eager_ms = _bench(eager_batch, args.warm, args.iters)
    fused_ms = _bench(fused_batch, args.warm, args.iters)
    print(f"batch step ({args.batch}x{args.t_slot} clips, ->bf16 CL):")
    print(f"  eager chain : {eager_ms:.4f} ms")
    print(f"  gather_cast : {fused_ms:.4f} ms  ({eager_ms / fused_ms:.2f}x)")

    x = ring.index_select(0, slot_t)[:, : args.t_ctx].reshape(-1, c, h, w).contiguous()

    def eager_cast():
        return x.to(torch.float32, memory_format=torch.channels_last).mul_(SCALE)

    def fused_cast():
        return scale_cast(x, SCALE, torch.bfloat16)

    e_ms = _bench(eager_cast, args.warm, args.iters)
    f_ms = _bench(fused_cast, args.warm, args.iters)
    print(f"scale_cast [2080,{c},{h},{w}] uint8 -> bf16 CL:")
    print(f"  eager to+mul_ : {e_ms:.4f} ms")
    print(f"  scale_cast    : {f_ms:.4f} ms  ({e_ms / f_ms:.2f}x)")


if __name__ == "__main__":
    main()
