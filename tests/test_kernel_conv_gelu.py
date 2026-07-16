import pytest
import torch

from pan2 import kernels
from pan2.models.encoder import FrameEncoder

CUDA_SHAPES = [
    ((64, 3, 64, 64), (32, 3, 7, 7), 3),
    ((64, 32, 32, 32), (64, 32, 3, 3), 1),
    ((2080, 3, 64, 64), (32, 3, 7, 7), 3),
    ((2080, 32, 32, 32), (64, 32, 3, 3), 1),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
@pytest.mark.parametrize("input_shape,weight_shape,padding", CUDA_SHAPES)
def test_conv_gelu_matches_fp32_reference(input_shape, weight_shape, padding):
    torch.manual_seed(0)
    x = (
        torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
        .mul_(0.02)
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_()
    )
    weight = (
        torch.randn(weight_shape, device="cuda", dtype=torch.bfloat16)
        .mul_(0.02)
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_()
    )
    actual = kernels.get("conv_gelu")(x, weight, 2, padding)
    grad = (
        torch.randn_like(actual)
        .mul_(1e-5)
        .contiguous(memory_format=torch.channels_last)
    )
    actual.backward(grad)

    x_ref = x.detach().float().requires_grad_()
    weight_ref = weight.detach().float().requires_grad_()
    expected = kernels.reference("conv_gelu")(x_ref, weight_ref, 2, padding)
    expected.backward(grad.float())

    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert x.grad is not None
    assert weight.grad is not None
    torch.testing.assert_close(actual.float(), expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(x.grad.float(), x_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        weight.grad.float(), weight_ref.grad, atol=1e-3, rtol=1e-3
    )


def test_conv_gelu_falls_back_for_unsupported_input():
    torch.manual_seed(1)
    x = torch.randn((3, 4, 17, 19), requires_grad=True)
    weight = torch.randn((7, 4, 3, 3), requires_grad=True)
    grad = torch.randn((3, 7, 9, 10))
    actual = kernels.get("conv_gelu")(x, weight, 2, 1)
    actual.backward(grad)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    expected = kernels.reference("conv_gelu")(x_ref, weight_ref, 2, 1)
    expected.backward(grad)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=0, rtol=0)


def test_encoder_frontend_parameter_names_are_unchanged():
    encoder = FrameEncoder(d_model=512, image_size=64, stem_channels=32)
    names = set(encoder.state_dict())
    assert "stem.conv.weight" in names
    assert "block1.b1_conv.weight" in names
    assert isinstance(encoder.stem.act, torch.nn.Identity)
    assert isinstance(encoder.block1.b1_act, torch.nn.Identity)
