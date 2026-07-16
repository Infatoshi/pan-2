"""Fused bias + GELU (erf) for transformer MLP intermediates.

Reference is pure PyTorch. Optimized path is a Triton elementwise kernel with
a custom autograd Function so forward and backward each do one memory pass
over the activation (bias broadcast + gelu / gelu' + bias grad reduce).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from pan2.kernels import register, register_reference

_SQRT_2_INV = 0.7071067811865476  # 1/sqrt(2)
_SQRT_2_PI_INV = 0.3989422804014327  # 1/sqrt(2*pi)


def bias_gelu_ref(x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Pure-torch reference: gelu(x + bias) with exact (erf) GELU."""
    if bias is not None:
        x = x + bias
    return F.gelu(x, approximate="none")


def _gelu_prime(u: torch.Tensor) -> torch.Tensor:
    """d/du gelu(u) for exact (erf) GELU, pure torch."""
    # gelu(u) = 0.5 * u * (1 + erf(u/sqrt(2)))
    # gelu'(u) = 0.5*(1+erf(u/sqrt(2))) + u * phi(u)
    # phi(u) = (1/sqrt(2*pi)) * exp(-0.5 u^2)
    cdf = 0.5 * (1.0 + torch.erf(u * _SQRT_2_INV))
    pdf = torch.exp(-0.5 * u * u) * _SQRT_2_PI_INV
    return cdf + u * pdf


def bias_gelu_bwd_ref(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference backward for bias_gelu_ref."""
    if bias is not None:
        u = x + bias
    else:
        u = x
    g = grad_out * _gelu_prime(u)
    db: torch.Tensor | None
    if bias is not None:
        # sum over all dims except last
        reduce_dims = tuple(range(g.ndim - 1))
        db = g.sum(dim=reduce_dims)
    else:
        db = None
    return g, db


# ---------------------------------------------------------------------------
# Triton path
# ---------------------------------------------------------------------------

_TRITON_OK = False
try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if _TRITON_OK:

    @triton.jit
    def _bias_gelu_fwd_kernel(
        x_ptr,
        bias_ptr,
        y_ptr,
        n_rows,
        n_cols,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        off = row * n_cols + cols
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            u = x + b
        else:
            u = x
        # exact gelu via erf
        y = 0.5 * u * (1.0 + tl.math.erf(u * 0.7071067811865476))
        tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _bias_gelu_bwd_kernel(
        go_ptr,
        x_ptr,
        bias_ptr,
        gx_ptr,
        n_rows,
        n_cols,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        off = row * n_cols + cols
        go = tl.load(go_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            u = x + b
        else:
            u = x
        # gelu'(u) = 0.5*(1+erf(u/sqrt(2))) + u * (1/sqrt(2*pi)) * exp(-0.5 u^2)
        cdf = 0.5 * (1.0 + tl.math.erf(u * 0.7071067811865476))
        pdf = tl.exp(-0.5 * u * u) * 0.3989422804014327
        gp = cdf + u * pdf
        gx = go * gp
        tl.store(gx_ptr + off, gx.to(gx_ptr.dtype.element_ty), mask=mask)

    def _launch_fwd(x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        x_c = x.contiguous()
        y = torch.empty_like(x_c)
        n_cols = x_c.shape[-1]
        n_rows = x_c.numel() // n_cols
        BLOCK = triton.next_power_of_2(n_cols)
        # cap block for large hidden (e.g. 2048) — still one program per row
        BLOCK = min(BLOCK, 4096)
        if BLOCK < n_cols:
            # rare: fall back to ref for huge last-dim
            return bias_gelu_ref(x_c, bias)
        bias_arg = bias.contiguous() if bias is not None else x_c  # dummy ptr
        _bias_gelu_fwd_kernel[(n_rows,)](
            x_c,
            bias_arg,
            y,
            n_rows,
            n_cols,
            HAS_BIAS=bias is not None,
            BLOCK=BLOCK,
        )
        return y

    def _launch_bwd(
        grad_out: torch.Tensor,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        go = grad_out.contiguous()
        x_c = x.contiguous()
        gx = torch.empty_like(x_c)
        n_cols = x_c.shape[-1]
        n_rows = x_c.numel() // n_cols
        BLOCK = min(triton.next_power_of_2(n_cols), 4096)
        if BLOCK < n_cols:
            g, _ = bias_gelu_bwd_ref(go, x_c, bias)
            return g
        bias_arg = bias.contiguous() if bias is not None else x_c
        _bias_gelu_bwd_kernel[(n_rows,)](
            go,
            x_c,
            bias_arg,
            gx,
            n_rows,
            n_cols,
            HAS_BIAS=bias is not None,
            BLOCK=BLOCK,
        )
        return gx


class _BiasGeluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        use_triton = (
            _TRITON_OK
            and x.is_cuda
            and x.numel() > 0
            and (bias is None or (bias.is_cuda and bias.shape[-1] == x.shape[-1]))
        )
        ctx.has_bias = bias is not None
        ctx.use_triton = use_triton
        if use_triton:
            # Save pre-activation inputs for Triton bwd (reconstructs u = x+bias).
            if bias is not None:
                ctx.save_for_backward(x, bias)
            else:
                ctx.save_for_backward(x)
            return _launch_fwd(x, bias)
        # CPU / no-triton: save u = x(+bias) for gelu'
        u = x + bias if bias is not None else x
        ctx.save_for_backward(u)
        return F.gelu(u, approximate="none")

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        if ctx.use_triton:
            if ctx.has_bias:
                x, bias = ctx.saved_tensors
            else:
                (x,) = ctx.saved_tensors
                bias = None
            gx = _launch_bwd(grad_out, x, bias)
            if ctx.has_bias:
                reduce_dims = tuple(range(gx.ndim - 1))
                db = gx.sum(dim=reduce_dims)
            else:
                db = None
            return gx, db
        (u,) = ctx.saved_tensors
        g = grad_out * _gelu_prime(u)
        if ctx.has_bias:
            reduce_dims = tuple(range(g.ndim - 1))
            db = g.sum(dim=reduce_dims)
        else:
            db = None
        return g, db


def bias_gelu(x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Best available bias+GELU (Triton on CUDA, else pure torch)."""
    return _BiasGeluFn.apply(x, bias)


# Register on import
register_reference("bias_gelu", bias_gelu_ref)
register("bias_gelu", bias_gelu, reference=bias_gelu_ref)
register_reference("bias_gelu_bwd", bias_gelu_bwd_ref)
