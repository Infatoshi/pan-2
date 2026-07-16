"""Fused channels-last convolution and exact GELU for the encoder frontend."""

from __future__ import annotations

import os

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


def conv_gelu_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """Pure-PyTorch convolution followed by exact GELU."""
    return F.gelu(F.conv2d(x, weight, stride=stride, padding=padding), approximate="none")


if _HAS_TRITON:

    @triton.jit
    def _conv_gelu_fwd_kernel(
        x_ptr,
        weight_ptr,
        y_ptr,
        pre_ptr,
        N,
        CIN: tl.constexpr,
        COUT: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        OH: tl.constexpr,
        OW: tl.constexpr,
        STRIDE: tl.constexpr,
        PADDING: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        output_rows = N * OH * OW
        n = offs_m // (OH * OW)
        output_spatial = offs_m % (OH * OW)
        oh = output_spatial // OW
        ow = output_spatial % OW
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        kernel_elements: tl.constexpr = KH * KW * CIN
        for k_base in range(0, kernel_elements, BLOCK_K):
            offs_k = k_base + tl.arange(0, BLOCK_K)
            kh = offs_k // (KW * CIN)
            kw = (offs_k // CIN) % KW
            channel = offs_k % CIN
            ih = oh[:, None] * STRIDE + kh[None, :] - PADDING
            iw = ow[:, None] * STRIDE + kw[None, :] - PADDING
            x_offsets = (
                n[:, None] * H * W * CIN
                + ih * W * CIN
                + iw * CIN
                + channel[None, :]
            )
            x_mask = (
                (offs_m[:, None] < output_rows)
                & (offs_k[None, :] < kernel_elements)
                & (ih >= 0)
                & (ih < H)
                & (iw >= 0)
                & (iw < W)
            )
            x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            weight_offsets = offs_k[:, None] + offs_n[None, :] * kernel_elements
            weight_mask = (offs_k[:, None] < kernel_elements) & (
                offs_n[None, :] < COUT
            )
            weight = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
            accumulator = tl.dot(x, weight, accumulator)

        physical = offs_m[:, None] * COUT + offs_n[None, :]
        output_mask = (offs_m[:, None] < output_rows) & (offs_n[None, :] < COUT)
        gelu = 0.5 * accumulator * (
            1.0 + tl.erf(accumulator * 0.7071067811865476)
        )
        tl.store(pre_ptr + physical, accumulator, mask=output_mask)
        tl.store(y_ptr + physical, gelu, mask=output_mask)

    @triton.jit
    def _conv_gelu_bwd_epilogue_kernel(
        grad_ptr,
        pre_ptr,
        dpre_ptr,
        elements,
        COUT: tl.constexpr,
        OH: tl.constexpr,
        OW: tl.constexpr,
        grad_stride_n,
        grad_stride_c,
        grad_stride_h,
        grad_stride_w,
        BLOCK: tl.constexpr,
    ):
        physical = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = physical < elements
        channel = physical % COUT
        output_row = physical // COUT
        n = output_row // (OH * OW)
        spatial = output_row % (OH * OW)
        oh = spatial // OW
        ow = spatial % OW
        grad_offset = (
            n * grad_stride_n
            + channel * grad_stride_c
            + oh * grad_stride_h
            + ow * grad_stride_w
        )
        grad = tl.load(grad_ptr + grad_offset, mask=mask, other=0.0).to(tl.float32)
        pre = tl.load(pre_ptr + physical, mask=mask, other=0.0).to(tl.float32)
        cdf = 0.5 * (1.0 + tl.erf(pre * 0.7071067811865476))
        pdf_term = pre * tl.exp(-0.5 * pre * pre) * 0.3989422804014327
        tl.store(dpre_ptr + physical, grad * (cdf + pdf_term), mask=mask)

    @triton.jit
    def _conv_gelu_stem_dgrad_packed_kernel(
        dpre_ptr,
        weight_ptr,
        dx_ptr,
        N,
        H: tl.constexpr,
        W: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        OH: tl.constexpr,
        OW: tl.constexpr,
        STRIDE: tl.constexpr,
        PADDING: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_COL: tl.constexpr,
        BLOCK_COUT: tl.constexpr,
    ):
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_col = tl.arange(0, BLOCK_COL)
        half_h: tl.constexpr = H // STRIDE
        half_w: tl.constexpr = W // STRIDE
        rows = N * half_h * half_w
        n = offs_m // (half_h * half_w)
        spatial = offs_m % (half_h * half_w)
        input_h_base = (spatial // half_w) * STRIDE
        input_w_base = (spatial % half_w) * STRIDE
        column_parity = offs_col // 3
        channel = offs_col % 3
        accumulator = tl.zeros((BLOCK_M, BLOCK_COL), dtype=tl.float32)
        offs_cout = tl.arange(0, BLOCK_COUT)
        kernel_elements: tl.constexpr = KH * KW * 3

        for kh in range(KH):
            parity_h = (kh - PADDING) % STRIDE
            oh = (input_h_base + parity_h + PADDING - kh) // STRIDE
            valid_h = (oh >= 0) & (oh < OH)
            for kw in range(KW):
                parity_w = (kw - PADDING) % STRIDE
                parity = parity_h * STRIDE + parity_w
                ow = (input_w_base + parity_w + PADDING - kw) // STRIDE
                output_row = n * OH * OW + oh * OW + ow
                dpre_mask = (
                    (offs_m[:, None] < rows)
                    & valid_h[:, None]
                    & (ow[:, None] >= 0)
                    & (ow[:, None] < OW)
                )
                dpre = tl.load(
                    dpre_ptr + output_row[:, None] * 32 + offs_cout[None, :],
                    mask=dpre_mask,
                    other=0.0,
                )
                weight_offsets = (
                    offs_cout[:, None] * kernel_elements
                    + kh * KW * 3
                    + kw * 3
                    + channel[None, :]
                )
                weight_mask = (offs_col[None, :] < 12) & (
                    column_parity[None, :] == parity
                )
                weight = tl.load(
                    weight_ptr + weight_offsets, mask=weight_mask, other=0.0
                )
                accumulator = tl.dot(dpre, weight, accumulator)

        parity_h = column_parity // STRIDE
        parity_w = column_parity % STRIDE
        input_offsets = (
            (
                (n[:, None] * H + input_h_base[:, None] + parity_h[None, :]) * W
                + input_w_base[:, None]
                + parity_w[None, :]
            )
            * 3
            + channel[None, :]
        )
        output_mask = (offs_m[:, None] < rows) & (offs_col[None, :] < 12)
        tl.store(dx_ptr + input_offsets, accumulator, mask=output_mask)

    @triton.jit
    def _conv_gelu_dgrad_kernel(
        dpre_ptr,
        weight_ptr,
        dx_ptr,
        N,
        CIN: tl.constexpr,
        COUT: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        OH: tl.constexpr,
        OW: tl.constexpr,
        STRIDE: tl.constexpr,
        PADDING: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_CIN: tl.constexpr,
        BLOCK_COUT: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        parity = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cin = tl.arange(0, BLOCK_CIN)
        half_h: tl.constexpr = H // STRIDE
        half_w: tl.constexpr = W // STRIDE
        input_rows = N * half_h * half_w
        n = offs_m // (half_h * half_w)
        input_spatial = offs_m % (half_h * half_w)
        ih = (input_spatial // half_w) * STRIDE + parity // STRIDE
        iw = (input_spatial % half_w) * STRIDE + parity % STRIDE
        accumulator = tl.zeros((BLOCK_M, BLOCK_CIN), dtype=tl.float32)
        kernel_elements: tl.constexpr = KH * KW * CIN

        # Parity-strided kh/kw: only filters that land on this input parity.
        # Trip count is ceil(K/STRIDE), which overshoots when start != 0
        # (KH=3,s=2,kh_start=1 -> kh=1,3). dpre was already masked for
        # kh>=KH, but weight was not: OOB weight loads read neighboring
        # device memory (often nan under allocator reuse) and 0*nan
        # poisoned tl.dot. Clamp the weight address into-bounds and mask
        # the load so invalid (kh,kw) contribute zero.
        kh_start = (parity // STRIDE + PADDING) % STRIDE
        kw_start = (parity % STRIDE + PADDING) % STRIDE
        for kh_index in range((KH + STRIDE - 1) // STRIDE):
            kh = kh_start + kh_index * STRIDE
            kh_safe = tl.minimum(kh, KH - 1)
            oh_numerator = ih + PADDING - kh
            oh = oh_numerator // STRIDE
            valid_h = (kh < KH) & (oh_numerator >= 0) & (oh < OH)
            for kw_index in range((KW + STRIDE - 1) // STRIDE):
                kw = kw_start + kw_index * STRIDE
                kw_safe = tl.minimum(kw, KW - 1)
                ow_numerator = iw + PADDING - kw
                ow = ow_numerator // STRIDE
                valid_spatial = (
                    valid_h
                    & (kw < KW)
                    & (ow_numerator >= 0)
                    & (ow < OW)
                )
                output_row = n * OH * OW + oh * OW + ow
                for cout_base in range(0, COUT, BLOCK_COUT):
                    offs_cout = cout_base + tl.arange(0, BLOCK_COUT)
                    output_mask = (
                        (offs_m[:, None] < input_rows)
                        & valid_spatial[:, None]
                        & (offs_cout[None, :] < COUT)
                    )
                    dpre = tl.load(
                        dpre_ptr + output_row[:, None] * COUT + offs_cout[None, :],
                        mask=output_mask,
                        other=0.0,
                    )
                    weight_offsets = (
                        offs_cout[:, None] * kernel_elements
                        + kh_safe * KW * CIN
                        + kw_safe * CIN
                        + offs_cin[None, :]
                    )
                    weight_mask = (
                        (offs_cout[:, None] < COUT)
                        & (offs_cin[None, :] < CIN)
                        & (kh < KH)
                        & (kw < KW)
                    )
                    weight = tl.load(
                        weight_ptr + weight_offsets, mask=weight_mask, other=0.0
                    )
                    accumulator = tl.dot(dpre, weight, accumulator)

        input_physical = (n * H + ih) * W + iw
        dx_mask = (offs_m[:, None] < input_rows) & (offs_cin[None, :] < CIN)
        tl.store(
            dx_ptr + input_physical[:, None] * CIN + offs_cin[None, :],
            accumulator,
            mask=dx_mask,
        )

    @triton.jit
    def _conv_gelu_wgrad_kernel(
        x_ptr,
        dpre_ptr,
        dweight_ptr,
        N,
        CIN: tl.constexpr,
        COUT: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        OH: tl.constexpr,
        OW: tl.constexpr,
        STRIDE: tl.constexpr,
        PADDING: tl.constexpr,
        SPLITS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_COUT: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_cout = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_split = tl.program_id(2)
        offs_cout = pid_cout * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        kernel_elements: tl.constexpr = KH * KW * CIN
        output_rows = N * OH * OW
        kh = offs_k // (KW * CIN)
        kw = (offs_k // CIN) % KW
        channel = offs_k % CIN
        accumulator = tl.zeros((BLOCK_COUT, BLOCK_K), dtype=tl.float32)

        for r_base in range(pid_split * BLOCK_R, output_rows, SPLITS * BLOCK_R):
            offs_r = r_base + tl.arange(0, BLOCK_R)
            n = offs_r // (OH * OW)
            output_spatial = offs_r % (OH * OW)
            oh = output_spatial // OW
            ow = output_spatial % OW
            dpre_mask = (offs_cout[:, None] < COUT) & (
                offs_r[None, :] < output_rows
            )
            dpre = tl.load(
                dpre_ptr + offs_r[None, :] * COUT + offs_cout[:, None],
                mask=dpre_mask,
                other=0.0,
            )

            ih = oh[:, None] * STRIDE + kh[None, :] - PADDING
            iw = ow[:, None] * STRIDE + kw[None, :] - PADDING
            x_offsets = (
                n[:, None] * H * W * CIN
                + ih * W * CIN
                + iw * CIN
                + channel[None, :]
            )
            x_mask = (
                (offs_r[:, None] < output_rows)
                & (offs_k[None, :] < kernel_elements)
                & (ih >= 0)
                & (ih < H)
                & (iw >= 0)
                & (iw < W)
            )
            x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            accumulator = tl.dot(dpre, x, accumulator)

        output_offsets = offs_cout[:, None] * kernel_elements + offs_k[None, :]
        output_mask = (offs_cout[:, None] < COUT) & (
            offs_k[None, :] < kernel_elements
        )
        tl.atomic_add(dweight_ptr + output_offsets, accumulator, mask=output_mask)


def _output_shape(x: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    kh, kw = weight.shape[2:]
    oh = (x.shape[2] + 2 * padding - kh) // stride + 1
    ow = (x.shape[3] + 2 * padding - kw) // stride + 1
    return x.shape[0], weight.shape[0], oh, ow


def _supported_shape(
    x: torch.Tensor, weight: torch.Tensor, stride: int, padding: int
) -> bool:
    shape = (tuple(x.shape[1:]), tuple(weight.shape), stride, padding)
    return shape in {
        ((3, 64, 64), (32, 3, 7, 7), 2, 3),
        ((32, 32, 32), (64, 32, 3, 3), 2, 1),
    }


def _env_triton_enabled() -> bool:
    # PAN2_CONV_GELU_TRITON default OFF. 2026-07-15 flake root cause (kF):
    # generic dgrad parity-strided kh loop used trip count ceil(K/STRIDE),
    # which overshoots when kh_start != 0 (KH=3,s=2,kh_start=1 -> kh=1,3).
    # dpre was masked for kh>=KH but weight was not; OOB weight loads read
    # neighboring device memory (often nan under allocator reuse) and
    # 0*nan poisoned tl.dot. Fixed by clamping weight addresses in-bounds
    # and masking loads for kh>=KH / kw>=KW. Keep default off until
    # acceptance re-runs flip it; opt in with =1.
    raw = os.environ.get("PAN2_CONV_GELU_TRITON")
    if raw is None:
        return False
    return raw.strip().lower() not in ("0", "false", "off", "no")


def _can_use_triton(
    x: torch.Tensor, weight: torch.Tensor, stride: int, padding: int
) -> bool:
    return bool(
        _env_triton_enabled()
        and _HAS_TRITON
        and x.is_cuda
        and weight.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous(memory_format=torch.channels_last)
        and weight.is_contiguous(memory_format=torch.channels_last)
        and _supported_shape(x, weight, stride, padding)
    )


class _ConvGelu(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        n, cin, h, w = x.shape
        cout, _, kh, kw = weight.shape
        _, _, oh, ow = _output_shape(x, weight, stride, padding)
        y = torch.empty(
            (n, cout, oh, ow),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last,
        )
        pre = torch.empty_like(y, memory_format=torch.channels_last)
        block_n = 32 if cin == 3 else 64
        block_m = 256 if cin == 3 else 64
        _conv_gelu_fwd_kernel[(triton.cdiv(n * oh * ow, block_m), triton.cdiv(cout, block_n))](
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
            PADDING=padding,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=32 if cin == 3 else 64,
            num_warps=8,
        )
        ctx.save_for_backward(x, weight, pre)
        ctx.stride = stride
        ctx.padding = padding
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, pre = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        n, cin, h, w = x.shape
        cout, _, kh, kw = weight.shape
        oh, ow = pre.shape[2:]
        need_dx, need_dweight = ctx.needs_input_grad[:2]
        dx = (
            torch.empty_like(x, memory_format=torch.channels_last)
            if need_dx
            else None
        )
        dpre = torch.empty_like(pre, memory_format=torch.channels_last)
        dweight = (
            torch.zeros_like(
                weight, dtype=torch.float32, memory_format=torch.channels_last
            )
            if need_dweight
            else None
        )
        elements = pre.numel()
        _conv_gelu_bwd_epilogue_kernel[(triton.cdiv(elements, 1024),)](
            grad_output,
            pre,
            dpre,
            elements,
            COUT=cout,
            OH=oh,
            OW=ow,
            grad_stride_n=grad_output.stride(0),
            grad_stride_c=grad_output.stride(1),
            grad_stride_h=grad_output.stride(2),
            grad_stride_w=grad_output.stride(3),
            BLOCK=1024,
            num_warps=8,
        )
        if need_dx and cin == 3:
            assert dx is not None
            block_m = 128
            _conv_gelu_stem_dgrad_packed_kernel[
                (triton.cdiv(n * (h // stride) * (w // stride), block_m),)
            ](
                dpre,
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
                PADDING=padding,
                BLOCK_M=block_m,
                BLOCK_COL=16,
                BLOCK_COUT=32,
                num_warps=8,
                num_stages=2,
            )
        elif need_dx:
            assert dx is not None
            block_m = 256
            _conv_gelu_dgrad_kernel[
                (
                    triton.cdiv(n * (h // stride) * (w // stride), block_m),
                    stride * stride,
                )
            ](
                dpre,
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
                PADDING=padding,
                BLOCK_M=block_m,
                BLOCK_CIN=32,
                BLOCK_COUT=32,
                num_warps=8,
                num_stages=2,
            )
        if need_dweight:
            assert dweight is not None
            max_splits = 128 if cin == 3 else 64
            splits = min(max_splits, triton.cdiv(n * oh * ow, 128))
            _conv_gelu_wgrad_kernel[
                (triton.cdiv(cout, 32), triton.cdiv(kh * kw * cin, 32), splits)
            ](
                x,
                dpre,
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
                PADDING=padding,
                SPLITS=splits,
                BLOCK_R=128,
                BLOCK_COUT=32,
                BLOCK_K=32,
                num_warps=8,
            )
        return dx, None if dweight is None else dweight.to(weight.dtype), None, None


def conv_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """Use Triton for supported bf16 channels-last tensors, else PyTorch."""
    if _can_use_triton(x, weight, stride, padding):
        return _ConvGelu.apply(x, weight, stride, padding)
    return conv_gelu_ref(x, weight, stride, padding)


register("conv_gelu", conv_gelu, reference=conv_gelu_ref)
