"""layer_norm_affine optimized vs pure-torch reference."""

from __future__ import annotations

import pytest
import torch

from pan2.kernels import get, reference


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_layer_norm_affine_matches_ref(device: str) -> None:
    torch.manual_seed(0)
    b, t, d = 4, 16, 512
    x = torch.randn(b, t, d, device=device, dtype=torch.float32)
    w = torch.randn(d, device=device, dtype=torch.float32)
    bias = torch.randn(d, device=device, dtype=torch.float32)
    ref = reference("layer_norm_affine")
    opt = get("layer_norm_affine")
    y_ref = ref(x, (d,), w, bias, 1e-5)
    y_opt = opt(x, (d,), w, bias, 1e-5)
    torch.testing.assert_close(y_opt, y_ref, atol=1e-5, rtol=1e-5)
