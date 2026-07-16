"""LayerNorm (affine) used by the temporal stack.

Reference is pure PyTorch F.layer_norm. Optimized path prefers
torch.nn.functional.layer_norm on CUDA (already a fused cudnn/ATen kernel)
and is registered so temporal code never hard-codes a backend. The win for
the elementwise bucket comes from pairing LN with residual fusion via
torch.compile on the temporal module (see temporal.py), not from reimplementing
LN in Triton.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pan2.kernels import register, register_reference


def layer_norm_affine_ref(
    x: torch.Tensor,
    normalized_shape: list[int] | torch.Size,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pure-torch LayerNorm with optional affine."""
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


def layer_norm_affine(
    x: torch.Tensor,
    normalized_shape: list[int] | torch.Size,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Best available LayerNorm (ATen fused on CUDA)."""
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


register_reference("layer_norm_affine", layer_norm_affine_ref)
register("layer_norm_affine", layer_norm_affine, reference=layer_norm_affine_ref)
