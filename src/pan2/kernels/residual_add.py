"""Residual add used by pre-norm transformer blocks.

Reference and optimized path are both pure `x + y`. At production residual
shapes (B=32, T=67, D=512) a Triton elementwise add is *slower* than ATen
(see `scripts/bench_residual_add.py`); keeping the op as a registered kernel
lets temporal call sites go through `kernels.get` so a faster backend can be
swapped in later without model changes. Under `torch.compile`, ATen add is
what inductor fuses into larger elementwise regions.
"""

from __future__ import annotations

import torch

from pan2.kernels import register, register_reference


def residual_add_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pure-torch residual: x + y."""
    return x + y


def residual_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Best available residual add (ATen add; inductor-fusible)."""
    return x + y


register_reference("residual_add", residual_add_ref)
register("residual_add", residual_add, reference=residual_add_ref)
