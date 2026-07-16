"""residual_add optimized vs pure-torch reference."""

from __future__ import annotations

import pytest
import torch

from pan2.kernels import get, reference

_SHAPES = (
    (32, 67, 512),
    (4, 16, 512),
    (2, 8, 128),
)


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("device", _devices())
def test_residual_add_fwd(shape: tuple[int, ...], device: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, device=device, dtype=torch.float32)
    y = torch.randn(*shape, device=device, dtype=torch.float32)
    ref = reference("residual_add")
    opt = get("residual_add")
    torch.testing.assert_close(opt(x, y), ref(x, y), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", _SHAPES[:1])
@pytest.mark.parametrize("device", _devices())
def test_residual_add_bwd(shape: tuple[int, ...], device: str) -> None:
    torch.manual_seed(1)
    x = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
    ref = reference("residual_add")
    opt = get("residual_add")

    x_r = x.detach().clone().requires_grad_(True)
    y_r = y.detach().clone().requires_grad_(True)
    x_o = x.detach().clone().requires_grad_(True)
    y_o = y.detach().clone().requires_grad_(True)

    ref(x_r, y_r).sum().backward()
    opt(x_o, y_o).sum().backward()
    torch.testing.assert_close(x_o.grad, x_r.grad, atol=0.0, rtol=0.0)
    torch.testing.assert_close(y_o.grad, y_r.grad, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_residual_add_bf16() -> None:
    torch.manual_seed(2)
    shape = (32, 67, 512)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    ref = reference("residual_add")
    opt = get("residual_add")
    # exact for add in bf16 when both paths use same dtype
    torch.testing.assert_close(opt(x, y), ref(x, y), atol=0.0, rtol=0.0)
