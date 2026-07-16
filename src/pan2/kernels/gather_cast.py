"""Fused uint8 -> scaled float channels-last cast, with optional ring gather.

Two ops sharing one idea: read uint8 NCHW once, scale in fp32 registers,
store float dtype in channels_last layout. Replaces the eager chain
`index_select -> reshape copy -> .to(float32, CL) -> mul_ -> autocast bf16
cast` (~600 MB of traffic per train step at production shape) with a single
pass (~100 MB).

- `scale_cast(x, scale, out_dtype)`: [N,C,H,W] uint8 -> float CL, valuewise
  identical to `x.to(float32).mul_(scale).to(out_dtype, channels_last)`.
- `gather_cast(ring, slot_idx, scale, out_dtype, t_ctx)`: gathers B slots
  from a [S,T,C,H,W] uint8 ring and emits ctx rows [B*t_ctx,C,H,W] and tail
  rows [B*(T-t_ctx),C,H,W] packed contiguous channels_last in one pass
  (tail slots are the baked goal/neg rows in the pretrain ring layout).

Both are bit-exact against their pure-torch references (uint8 -> fp32 mul ->
single rounding at the destination dtype) and carry no autograd (uint8
inputs never require grad).
"""

from __future__ import annotations

import torch

from pan2.kernels import register

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only by CPU-only torch wheels
    triton = None
    tl = None
    _HAS_TRITON = False


def scale_cast_ref(x: torch.Tensor, scale: float, out_dtype: torch.dtype) -> torch.Tensor:
    """Pure-PyTorch uint8 -> scaled float channels-last cast (any ndim >= 4)."""
    lead = x.shape[:-3]
    x2 = x.reshape(-1, *x.shape[-3:])
    y2 = (
        x2.to(torch.float32)
        .mul_(scale)
        .to(dtype=out_dtype, memory_format=torch.channels_last)
    )
    return y2.reshape(*lead, *y2.shape[1:])


def gather_cast_ref(
    ring: torch.Tensor,
    slot_idx: torch.Tensor,
    scale: float,
    out_dtype: torch.dtype,
    t_ctx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch gather of B ring slots, split into packed ctx/tail rows."""
    b = slot_idx.shape[0]
    t = ring.shape[1]
    y = scale_cast_ref(ring.index_select(0, slot_idx), scale, out_dtype)
    ctx = y[:, :t_ctx].reshape(b * t_ctx, *y.shape[2:])
    tail = y[:, t_ctx:].reshape(b * (t - t_ctx), *y.shape[2:])
    return ctx, tail


if _HAS_TRITON:
    @triton.jit
    def _scale_cast_kernel(
        x_ptr,
        dst_ptr,
        n_elements,
        SCALE: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        HWC: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        n = offs // HWC
        within = offs % HWC
        spatial = within // C
        c = within % C
        src = n * HWC + c * HW + spatial  # NCHW contiguous source
        v = tl.load(x_ptr + src, mask=mask, other=0)
        y = v.to(tl.float32) * SCALE
        tl.store(dst_ptr + offs, y.to(dst_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _gather_cast_kernel(
        ring_ptr,
        idx_ptr,
        dst_ptr,
        n_ctx_rows,
        SCALE: tl.constexpr,
        T: tl.constexpr,
        T_CTX: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        HWC: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        b = row // T
        t = row % T
        slot = tl.load(idx_ptr + b)
        src_base = (slot * T + t) * HWC
        # pack ctx rows of all batches first, then tail rows
        n_tail = T - T_CTX
        dst_row = tl.where(t < T_CTX, b * T_CTX + t, n_ctx_rows + b * n_tail + (t - T_CTX))
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < HWC
        spatial = offs // C
        c = offs % C
        v = tl.load(ring_ptr + src_base + c * HW + spatial, mask=mask, other=0)
        y = v.to(tl.float32) * SCALE
        tl.store(dst_ptr + dst_row * HWC + offs, y.to(dst_ptr.dtype.element_ty), mask=mask)


def _can_use_triton(x: torch.Tensor) -> bool:
    return (
        _HAS_TRITON
        and x.is_cuda
        and x.dtype == torch.uint8
        and x.is_contiguous()
        and x.ndim >= 4
    )


def _block(hwc: int) -> int:
    return min(triton.next_power_of_2(hwc), 4096)


def scale_cast(x: torch.Tensor, scale: float, out_dtype: torch.dtype) -> torch.Tensor:
    """Use Triton for contiguous uint8 CUDA tensors, else PyTorch."""
    if not _can_use_triton(x):
        return scale_cast_ref(x, scale, out_dtype)
    c, h, w = x.shape[-3:]
    lead = x.shape[:-3]
    x2 = x.reshape(-1, c, h, w)
    dst = torch.empty(
        x2.shape, dtype=out_dtype, device=x.device,
        memory_format=torch.channels_last,
    )
    hw = h * w
    hwc = hw * c
    total = x2.numel()
    _scale_cast_kernel[(triton.cdiv(total, _block(hwc)),)](
        x2,
        dst,
        total,
        SCALE=scale,
        C=c,
        HW=hw,
        HWC=hwc,
        BLOCK=_block(hwc),
        num_warps=8,
    )
    return dst.reshape(*lead, c, h, w)


def gather_cast(
    ring: torch.Tensor,
    slot_idx: torch.Tensor,
    scale: float,
    out_dtype: torch.dtype,
    t_ctx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Triton for contiguous uint8 CUDA rings, else PyTorch."""
    if not (_can_use_triton(ring) and slot_idx.is_cuda):
        return gather_cast_ref(ring, slot_idx, scale, out_dtype, t_ctx)
    s, t, c, h, w = ring.shape
    b = slot_idx.shape[0]
    hw = h * w
    hwc = hw * c
    dst = torch.empty(
        (b * t, c, h, w), dtype=out_dtype, device=ring.device,
        memory_format=torch.channels_last,
    )
    n_ctx_rows = b * t_ctx
    _gather_cast_kernel[(b * t, triton.cdiv(hwc, _block(hwc)))](
        ring,
        slot_idx,
        dst,
        n_ctx_rows,
        SCALE=scale,
        T=t,
        T_CTX=t_ctx,
        C=c,
        HW=hw,
        HWC=hwc,
        BLOCK=_block(hwc),
        num_warps=8,
    )
    ctx = dst[:n_ctx_rows]
    tail = dst[n_ctx_rows:]
    # slices along dim0 keep channels-last strides and metadata
    return ctx, tail


register("scale_cast", scale_cast, reference=scale_cast_ref)
register("gather_cast", gather_cast, reference=gather_cast_ref)
