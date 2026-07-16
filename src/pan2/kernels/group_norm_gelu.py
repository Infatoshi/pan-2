"""Fused GroupNorm with affine transform and exact GELU."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pan2.kernels import register

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only by CPU-only torch wheels
    triton = None
    tl = None
    _HAS_TRITON = False


def group_norm_gelu_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pure-PyTorch GroupNorm followed by exact GELU."""
    return F.gelu(F.group_norm(x, num_groups, weight, bias, eps), approximate="none")


if _HAS_TRITON:
    @triton.jit
    def _group_norm_gelu_fwd_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        C: tl.constexpr,
        HW: tl.constexpr,
        CHANNELS_PER_GROUP: tl.constexpr,
        ELEMENTS_PER_GROUP: tl.constexpr,
        GROUPS: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        n = row // GROUPS
        group = row % GROUPS
        offsets = tl.arange(0, BLOCK)
        mask = offsets < ELEMENTS_PER_GROUP
        spatial = offsets // CHANNELS_PER_GROUP
        channel = group * CHANNELS_PER_GROUP + offsets % CHANNELS_PER_GROUP
        physical = n * HW * C + spatial * C + channel

        x = tl.load(x_ptr + physical, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / ELEMENTS_PER_GROUP
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / ELEMENTS_PER_GROUP
        rstd = tl.rsqrt(variance + EPS)
        normalized = centered * rstd
        affine = normalized * tl.load(weight_ptr + channel, mask=mask, other=0.0)
        affine += tl.load(bias_ptr + channel, mask=mask, other=0.0)
        y = 0.5 * affine * (1.0 + tl.erf(affine * 0.7071067811865476))

        tl.store(y_ptr + physical, y, mask=mask)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _group_norm_gelu_bwd_kernel(
        grad_ptr,
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        rstd_ptr,
        dx_ptr,
        dweight_ptr,
        dbias_ptr,
        grad_stride_n: tl.constexpr,
        grad_stride_c: tl.constexpr,
        grad_stride_h: tl.constexpr,
        grad_stride_w: tl.constexpr,
        C: tl.constexpr,
        W: tl.constexpr,
        HW: tl.constexpr,
        CHANNELS_PER_GROUP: tl.constexpr,
        ELEMENTS_PER_GROUP: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        n = row // GROUPS
        group = row % GROUPS
        offsets = tl.arange(0, BLOCK)
        mask = offsets < ELEMENTS_PER_GROUP
        spatial = offsets // CHANNELS_PER_GROUP
        channel_local = offsets % CHANNELS_PER_GROUP
        channel = group * CHANNELS_PER_GROUP + channel_local
        physical = n * HW * C + spatial * C + channel
        h = spatial // W
        w = spatial % W
        grad_physical = (
            n * grad_stride_n
            + channel * grad_stride_c
            + h * grad_stride_h
            + w * grad_stride_w
        )

        x = tl.load(x_ptr + physical, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + grad_physical, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        normalized = (x - mean) * rstd
        affine = normalized * weight + tl.load(bias_ptr + channel, mask=mask, other=0.0)
        cdf = 0.5 * (1.0 + tl.erf(affine * 0.7071067811865476))
        pdf_term = affine * tl.exp(-0.5 * affine * affine) * 0.3989422804014327
        grad_affine = grad * (cdf + pdf_term)

        grad_normalized = grad_affine * weight
        sum_grad = tl.sum(tl.where(mask, grad_normalized, 0.0), axis=0)
        sum_grad_norm = tl.sum(
            tl.where(mask, grad_normalized * normalized, 0.0), axis=0
        )
        dx = (
            rstd
            / ELEMENTS_PER_GROUP
            * (ELEMENTS_PER_GROUP * grad_normalized - sum_grad - normalized * sum_grad_norm)
        )
        tl.store(dx_ptr + physical, dx, mask=mask)

        grad_affine_2d = tl.reshape(grad_affine, (HW, CHANNELS_PER_GROUP))
        normalized_2d = tl.reshape(normalized, (HW, CHANNELS_PER_GROUP))
        dweight = tl.sum(grad_affine_2d * normalized_2d, axis=0)
        dbias = tl.sum(grad_affine_2d, axis=0)
        channel_offsets = group * CHANNELS_PER_GROUP + tl.arange(0, CHANNELS_PER_GROUP)
        tl.atomic_add(dweight_ptr + channel_offsets, dweight)
        tl.atomic_add(dbias_ptr + channel_offsets, dbias)


class _GroupNormGelu(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        num_groups: int,
        eps: float,
    ) -> torch.Tensor:
        n, c, h, w = x.shape
        channels_per_group = c // num_groups
        elements_per_group = channels_per_group * h * w
        y = torch.empty_like(x, memory_format=torch.channels_last)
        mean = torch.empty((n * num_groups,), device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        _group_norm_gelu_fwd_kernel[(n * num_groups,)](
            x,
            weight,
            bias,
            y,
            mean,
            rstd,
            C=c,
            HW=h * w,
            CHANNELS_PER_GROUP=channels_per_group,
            ELEMENTS_PER_GROUP=elements_per_group,
            GROUPS=num_groups,
            EPS=eps,
            BLOCK=triton.next_power_of_2(elements_per_group),
            num_warps=8,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.num_groups = num_groups
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        n, c, h, w = x.shape
        num_groups = ctx.num_groups
        channels_per_group = c // num_groups
        elements_per_group = channels_per_group * h * w
        dx = torch.empty_like(x, memory_format=torch.channels_last)
        dweight = torch.zeros_like(weight, dtype=torch.float32)
        dbias = torch.zeros_like(bias, dtype=torch.float32)
        _group_norm_gelu_bwd_kernel[(n * num_groups,)](
            grad_output,
            x,
            weight,
            bias,
            mean,
            rstd,
            dx,
            dweight,
            dbias,
            grad_stride_n=grad_output.stride(0),
            grad_stride_c=grad_output.stride(1),
            grad_stride_h=grad_output.stride(2),
            grad_stride_w=grad_output.stride(3),
            C=c,
            W=w,
            HW=h * w,
            CHANNELS_PER_GROUP=channels_per_group,
            ELEMENTS_PER_GROUP=elements_per_group,
            GROUPS=num_groups,
            BLOCK=triton.next_power_of_2(elements_per_group),
            num_warps=8,
        )
        return dx, dweight.to(weight.dtype), dbias.to(bias.dtype), None, None


def _can_use_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> bool:
    if not _HAS_TRITON or not x.is_cuda or x.ndim != 4:
        return False
    if not x.is_contiguous(memory_format=torch.channels_last):
        return False
    if weight is None or bias is None or x.shape[1] % 8 != 0:
        return False
    elements_per_group = (x.shape[1] // 8) * x.shape[2] * x.shape[3]
    return elements_per_group == triton.next_power_of_2(elements_per_group)


def group_norm_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Use Triton for supported channels-last CUDA tensors, else PyTorch."""
    if num_groups == 8 and _can_use_triton(x, weight, bias):
        return _GroupNormGelu.apply(x, weight, bias, num_groups, eps)
    return group_norm_gelu_ref(x, weight, bias, num_groups, eps)


register("group_norm_gelu", group_norm_gelu, reference=group_norm_gelu_ref)
