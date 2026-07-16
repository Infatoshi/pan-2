"""bias_gelu optimized vs pure-torch reference."""

from __future__ import annotations

import pytest
import torch

from pan2.kernels import get, reference

# Production MLP intermediate: B=32, T=67, H=2048 (d=512, ratio=4)
_SHAPES = (
    (32, 67, 2048),
    (4, 16, 512),
    (1, 8, 256),
)


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("device", _devices())
def test_bias_gelu_fwd_fp32(shape: tuple[int, ...], device: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, device=device, dtype=torch.float32)
    bias = torch.randn(shape[-1], device=device, dtype=torch.float32)
    ref = reference("bias_gelu")
    opt = get("bias_gelu")
    y_ref = ref(x, bias)
    y_opt = opt(x, bias)
    torch.testing.assert_close(y_opt, y_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", _SHAPES[:1])
@pytest.mark.parametrize("device", _devices())
def test_bias_gelu_fwd_no_bias(shape: tuple[int, ...], device: str) -> None:
    torch.manual_seed(1)
    x = torch.randn(*shape, device=device, dtype=torch.float32)
    ref = reference("bias_gelu")
    opt = get("bias_gelu")
    torch.testing.assert_close(opt(x, None), ref(x, None), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", _SHAPES[:2])
@pytest.mark.parametrize("device", _devices())
def test_bias_gelu_bwd_fp32(shape: tuple[int, ...], device: str) -> None:
    torch.manual_seed(2)
    x = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(shape[-1], device=device, dtype=torch.float32, requires_grad=True)
    ref = reference("bias_gelu")
    opt = get("bias_gelu")

    x_r = x.detach().clone().requires_grad_(True)
    b_r = bias.detach().clone().requires_grad_(True)
    x_o = x.detach().clone().requires_grad_(True)
    b_o = bias.detach().clone().requires_grad_(True)

    ref(x_r, b_r).square().mean().backward()
    opt(x_o, b_o).square().mean().backward()

    torch.testing.assert_close(x_o.grad, x_r.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(b_o.grad, b_r.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16 kernel path")
@pytest.mark.parametrize("shape", _SHAPES[:1])
def test_bias_gelu_bf16_matches_fp32_ref(shape: tuple[int, ...]) -> None:
    """Optimized bf16 vs pure-torch fp32 reference (atol/rtol 1e-3 on float view).

    Compare against the bf16-rounded fp32 reference so storage quantisation is
    shared; residual error is Triton vs ATen gelu in fp32 intermediates.
    """
    torch.manual_seed(3)
    device = "cuda"
    x_bf = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    b_bf = torch.randn(shape[-1], device=device, dtype=torch.bfloat16)
    ref = reference("bias_gelu")
    opt = get("bias_gelu")
    y_ref = ref(x_bf.float(), b_bf.float()).to(torch.bfloat16)
    y_opt = opt(x_bf, b_bf)
    torch.testing.assert_close(y_opt.float(), y_ref.float(), atol=1e-3, rtol=1e-3)
