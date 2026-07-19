#!/usr/bin/env python3
"""Offline block-config tuner for the conv_gelu kernel family (GPU0, sm_120).

Sweeps BLOCK/num_warps/num_stages variants per kernel at PRODUCTION shapes
(bs256 x 133 images = 34,048: stem, b1, b2.pw, b3.pw) and prints the best
configs per kernel+shape. Winners are hardcoded in kernels/conv_gelu.py -
no runtime autotune in production.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import triton
import triton.testing

from pan2.kernels.conv_gelu import (
    _conv_gelu_bwd_epilogue_kernel,
    _conv_gelu_dgrad_kernel,
    _conv_gelu_fwd_kernel,
    _conv_gelu_stem_dgrad_packed_kernel,
    _conv_gelu_wgrad_kernel,
)

DEV = "cuda"

# name, x shape, weight shape, stride, padding
# Only shapes whitelisted in _supported_shape (kernels/conv_gelu.py:401) run
# the Triton path in production; b2pw/b3pw 1x1 pointwise convs use the torch
# ref path, so they are not tuned here.
SHAPES = {
    "stem": ((34048, 3, 64, 64), (32, 3, 7, 7), 2, 3),
    "b1": ((34048, 32, 32, 32), (64, 32, 3, 3), 2, 1),
}


def out_hw(h, kh, stride, pad):
    return (h + 2 * pad - kh) // stride + 1


SMOKE = "--smoke" in sys.argv
BAD: set = set()


def bench(fn, key, warmup=10, rep=30):
    fn()  # compile + first launch; under --smoke (pair with
    # CUDA_LAUNCH_BLOCKING=1) an illegal access raises here
    torch.cuda.synchronize()
    if SMOKE:
        print(f"  SMOKE OK {key}", flush=True)
        return 0.0
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def badfile_load():
    import json
    import os

    if os.path.exists("/tmp/tune_bad.json"):
        return set(tuple(x) for x in json.load(open("/tmp/tune_bad.json")))
    return set()


def badfile_save():
    import json

    json.dump(sorted(list(x) for x in (BAD | KNOWN_BAD)), open("/tmp/tune_bad.json", "w"))


def tune_fwd(name, xs, ws, stride, pad, variants):
    n, cin, h, w = xs
    cout, _, kh, kw = ws
    oh, ow = out_hw(h, kh, stride, pad), out_hw(w, kw, stride, pad)
    x = torch.empty(
        xs, device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    weight = torch.empty(
        ws, device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    y = torch.empty(
        (n, cout, oh, ow), device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    )
    pre = torch.empty_like(y)
    rows = []
    for bm, bn, bk, nw, ns in variants:
        key = ("fwd", name, bm, bn, bk, nw, ns)
        if key in KNOWN_BAD:
            continue

        def go():
            _conv_gelu_fwd_kernel[(triton.cdiv(n * oh * ow, bm), triton.cdiv(cout, bn))](
                x,
                weight,
                y,
                pre,
                n,
                CIN=cin,
                COUT=cout,
                H=h,
                W=w,
                KH=kh,
                KW=kw,
                OH=oh,
                OW=ow,
                STRIDE=stride,
                PADDING=pad,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                num_warps=nw,
                num_stages=ns,
            )

        try:
            ms = bench(go, key)
        except Exception:
            BAD.add(key)
            if not SMOKE:
                raise  # timing pass must never hit a known-good-config crash
            badfile_save()
            print(f"SMOKE CRASH {key}", flush=True)
            sys.exit(3)
        rows.append((ms, dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns)))
    rows.sort(key=lambda r: r[0])
    if not rows:
        print("  (no rows)")
        return rows
    print(f"FWD {name}: best {rows[0][0]:.3f} ms {rows[0][1]}")
    for ms, cfg in rows[1:4]:
        print(f"      {ms:.3f} ms {cfg}")
    return rows


def tune_epilogue(name, xs, ws, stride, pad):
    n, cin, h, w = xs
    cout, _, kh, kw = ws
    oh, ow = out_hw(h, kh, stride, pad), out_hw(w, kw, stride, pad)
    pre = torch.empty(
        (n, cout, oh, ow), device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    grad = torch.randn_like(pre, memory_format=torch.channels_last)
    dpre = torch.empty_like(pre)
    elements = pre.numel()
    rows = []
    for blk, nw in itertools.product((512, 1024, 2048, 4096, 8192), (4, 8, 16)):
        key = ("epi", name, blk, nw)
        if key in KNOWN_BAD:
            continue

        def go():
            _conv_gelu_bwd_epilogue_kernel[(triton.cdiv(elements, blk),)](
                grad,
                pre,
                dpre,
                elements,
                COUT=cout,
                OH=oh,
                OW=ow,
                grad_stride_n=grad.stride(0),
                grad_stride_c=grad.stride(1),
                grad_stride_h=grad.stride(2),
                grad_stride_w=grad.stride(3),
                BLOCK=blk,
                num_warps=nw,
            )

        try:
            ms = bench(go, key)
        except Exception:
            BAD.add(key)
            if not SMOKE:
                raise
            badfile_save()
            print(f"SMOKE CRASH {key}", flush=True)
            sys.exit(3)
        rows.append((ms, dict(BLOCK=blk, num_warps=nw)))
    rows.sort(key=lambda r: r[0])
    if not rows:
        print("  (no rows)")
        return rows
    print(f"EPI {name}: best {rows[0][0]:.3f} ms {rows[0][1]}  (worst {rows[-1][0]:.3f})")
    return rows


def _dx_ref(dpre_in, weight, n, cin, h, w, stride, pad):
    """fp32 conv2d-autograd reference for dgrad correctness gating."""
    import torch.nn.functional as F

    xr = torch.zeros((n, cin, h, w), device=DEV, dtype=torch.float32, requires_grad=True)
    y = F.conv2d(xr, weight.float(), stride=stride, padding=pad)
    y.backward(dpre_in.float())
    return xr.grad.detach()


def _ok(dx, ref, atol=5e-3, rtol=5e-3):
    return torch.allclose(dx.float(), ref, atol=atol, rtol=rtol)


def tune_dgrad(name, xs, ws, stride, pad, variants):
    n, cin, h, w = xs
    cout, _, kh, kw = ws
    oh, ow = out_hw(h, kh, stride, pad), out_hw(w, kw, stride, pad)
    dpre_in = torch.empty(
        (n, cout, oh, ow), device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    weight = torch.empty(
        ws, device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    dx = torch.empty(
        (n, cin, h, w), device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    )
    ref = _dx_ref(dpre_in, weight, n, cin, h, w, stride, pad)
    rows = []
    for bm, bci, bco, nw, ns in variants:
        key = ("dg", name, bm, bci, bco, nw, ns)
        if key in KNOWN_BAD:
            continue

        def go():
            _conv_gelu_dgrad_kernel[
                (triton.cdiv(n * (h // stride) * (w // stride), bm), stride * stride)
            ](
                dpre_in,
                weight,
                dx,
                n,
                CIN=cin,
                COUT=cout,
                H=h,
                W=w,
                KH=kh,
                KW=kw,
                OH=oh,
                OW=ow,
                STRIDE=stride,
                PADDING=pad,
                BLOCK_M=bm,
                BLOCK_CIN=bci,
                BLOCK_COUT=bco,
                num_warps=nw,
                num_stages=ns,
            )

        try:
            ms = bench(go, key)
        except Exception:
            BAD.add(key)
            if not SMOKE:
                raise
            badfile_save()
            print(f"SMOKE CRASH {key}", flush=True)
            sys.exit(3)
        dg = "dg" if ws[1] != 3 else "sdg"
        if not _ok(dx, ref):
            print(f"  WRONG {dg} {key}", flush=True)
            continue
        rows.append(
            (ms, dict(BLOCK_M=bm, BLOCK_CIN=bci, BLOCK_COUT=bco, num_warps=nw, num_stages=ns))
        )
    rows.sort(key=lambda r: r[0])
    if not rows:
        print("  (no rows)")
        return rows
    print(f"DGRAD {name}: best {rows[0][0]:.3f} ms {rows[0][1]}  (worst {rows[-1][0]:.3f})")
    return rows


def tune_stem_dgrad(xs, ws, stride, pad, variants):
    n, cin, h, w = xs
    cout, _, kh, kw = ws
    oh, ow = out_hw(h, kh, stride, pad), out_hw(w, kw, stride, pad)
    dpre_in = torch.empty(
        (n, cout, oh, ow), device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    weight = torch.empty(
        ws, device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    dx = torch.empty(
        (n, cin, h, w), device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    )
    ref = _dx_ref(dpre_in, weight, n, cin, h, w, stride, pad)
    rows = []
    for bm, bcol, bco, nw, ns in variants:
        key = ("sdg", "stem", bm, bcol, bco, nw, ns)
        if key in KNOWN_BAD:
            continue

        def go():
            _conv_gelu_stem_dgrad_packed_kernel[
                (triton.cdiv(n * (h // stride) * (w // stride), bm),)
            ](
                dpre_in,
                weight,
                dx,
                n,
                H=h,
                W=w,
                KH=kh,
                KW=kw,
                OH=oh,
                OW=ow,
                STRIDE=stride,
                PADDING=pad,
                BLOCK_M=bm,
                BLOCK_COL=bcol,
                BLOCK_COUT=bco,
                num_warps=nw,
                num_stages=ns,
            )

        try:
            ms = bench(go, key)
        except Exception:
            BAD.add(key)
            if not SMOKE:
                raise
            badfile_save()
            print(f"SMOKE CRASH {key}", flush=True)
            sys.exit(3)
        if not _ok(dx, ref):
            print(f"  WRONG sdg {key}", flush=True)
            continue
        rows.append(
            (ms, dict(BLOCK_M=bm, BLOCK_COL=bcol, BLOCK_COUT=bco, num_warps=nw, num_stages=ns))
        )
    rows.sort(key=lambda r: r[0])
    if not rows:
        print("  (no rows)")
        return rows
    print(f"STEM-DGRAD: best {rows[0][0]:.3f} ms {rows[0][1]}  (worst {rows[-1][0]:.3f})")
    return rows


def tune_wgrad(name, xs, ws, stride, pad, variants):
    n, cin, h, w = xs
    cout, _, kh, kw = ws
    oh, ow = out_hw(h, kh, stride, pad), out_hw(w, kw, stride, pad)
    x = torch.empty(
        xs, device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    dpre_in = torch.empty(
        (n, cout, oh, ow), device=DEV, dtype=torch.bfloat16, memory_format=torch.channels_last
    ).normal_()
    dweight = torch.zeros(ws, device=DEV, dtype=torch.float32).to(memory_format=torch.channels_last)
    rows = []
    for br, bco, bk, maxsp, nw in variants:
        key = ("wg", name, br, bco, bk, maxsp, nw)
        if key in KNOWN_BAD:
            continue

        def go():
            splits = min(maxsp, triton.cdiv(n * oh * ow, 128))
            _conv_gelu_wgrad_kernel[
                (triton.cdiv(cout, bco), triton.cdiv(kh * kw * cin, bk), splits)
            ](
                x,
                dpre_in,
                dweight,
                n,
                CIN=cin,
                COUT=cout,
                H=h,
                W=w,
                KH=kh,
                KW=kw,
                OH=oh,
                OW=ow,
                STRIDE=stride,
                PADDING=pad,
                SPLITS=splits,
                BLOCK_R=br,
                BLOCK_COUT=bco,
                BLOCK_K=bk,
                num_warps=nw,
            )

        try:
            ms = bench(go, key)
        except Exception:
            BAD.add(key)
            if not SMOKE:
                raise
            badfile_save()
            print(f"SMOKE CRASH {key}", flush=True)
            sys.exit(3)
        rows.append(
            (ms, dict(BLOCK_R=br, BLOCK_COUT=bco, BLOCK_K=bk, max_splits=maxsp, num_warps=nw))
        )
    rows.sort(key=lambda r: r[0])
    if not rows:
        print("  (no rows)")
        return rows
    print(f"WGRAD {name}: best {rows[0][0]:.3f} ms {rows[0][1]}  (worst {rows[-1][0]:.3f})")
    return rows


KNOWN_BAD = badfile_load()


def main() -> None:
    torch.manual_seed(0)
    fwd_variants = [
        (bm, bn, bk, nw, ns)
        for bm in (64, 128, 256)
        for bn in (32, 64, 128)
        for bk in (32, 64)
        for nw in (4, 8)
        for ns in (2, 3, 4)
        if bm >= 16 and bn >= 16 and bk >= 16
    ]
    dg_variants = [
        (bm, bci, bco, nw, ns)
        for bm in (128, 256, 512)
        for bci in (16, 32, 64)
        for bco in (32, 64)
        for nw in (4, 8, 16)
        for ns in (1, 2, 3)
    ]
    stem_variants = [
        (bm, bcol, bco, nw, ns)
        for bm in (64, 128, 256)
        for bcol in (8, 16, 32)
        for bco in (16, 32, 64)
        for nw in (4, 8, 16)
        for ns in (1, 2, 3)
    ]
    wg_variants = [
        (br, bco, bk, maxsp, nw)
        for br in (64, 128, 256)
        for bco in (32, 64)
        for bk in (32, 64)
        for maxsp in (32, 64, 128, 256)
        for nw in (4, 8)
    ]
    for name, (xs, ws, stride, pad) in SHAPES.items():
        print(f"=== {name} x={xs} w={ws} s={stride}")
        cout, cin = ws[0], ws[1]
        k_elems = ws[2] * ws[3] * cin
        # prune pointless/oversized blocks (BLOCK_* > the dim they tile) - these
        # also trip an unmasked weight load in the packed stem dgrad
        fwd_v = [v for v in fwd_variants if v[1] <= cout and v[2] <= k_elems]
        dg_v = [v for v in dg_variants if v[1] <= cin and v[2] <= cout]
        sdg_v = [v for v in stem_variants if v[2] <= cout]
        wg_v = [v for v in wg_variants if v[1] <= cout and v[2] <= k_elems]
        tune_fwd(name, xs, ws, stride, pad, fwd_v)
        tune_epilogue(name, xs, ws, stride, pad)
        if stride == 2 and ws[1] == 3:
            tune_stem_dgrad(xs, ws, stride, pad, sdg_v)
        else:
            # generic dgrad covers every non-stem site, stride 1 and 2 both
            tune_dgrad(name, xs, ws, stride, pad, dg_v)
        tune_wgrad(name, xs, ws, stride, pad, wg_v)
        sys.stdout.flush()
    if SMOKE:
        badfile_save()
        print(f"SMOKE DONE: {len(BAD)} bad configs -> /tmp/tune_bad.json")


if __name__ == "__main__":
    main()
