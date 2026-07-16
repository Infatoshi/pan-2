import pytest
import torch

from pan2 import kernels
from pan2.models.encoder import FrameEncoder
from pan2.models.preprocess import prepare_images


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
@pytest.mark.parametrize("channels,height,width", [(128, 8, 8), (512, 4, 4)])
def test_group_norm_gelu_matches_fp32_reference(channels, height, width):
    torch.manual_seed(0)
    shape = (4, channels, height, width)
    x = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_()
    )
    weight = torch.full((channels,), 0.01, device="cuda", requires_grad=True)
    bias = torch.full((channels,), 0.02, device="cuda", requires_grad=True)
    grad = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        .mul_(0.01)
        .contiguous(memory_format=torch.channels_last)
    )

    actual = kernels.get("group_norm_gelu")(x, weight, bias, 8, 1e-5)
    actual.backward(grad)

    x_ref = x.detach().float().requires_grad_()
    weight_ref = weight.detach().float().requires_grad_()
    bias_ref = bias.detach().float().requires_grad_()
    expected = kernels.reference("group_norm_gelu")(
        x_ref, weight_ref, bias_ref, 8, 1e-5
    )
    expected.backward(grad.float())

    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(actual.float(), expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(x.grad.float(), x_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(weight.grad.float(), weight_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(bias.grad.float(), bias_ref.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
@pytest.mark.parametrize("channels,height,width", [(128, 8, 8), (512, 4, 4)])
def test_group_norm_gelu_fp32_random_affine(channels, height, width):
    torch.manual_seed(1)
    shape = (4, channels, height, width)
    x = (
        torch.randn(shape, device="cuda")
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_()
    )
    weight = torch.randn(channels, device="cuda", requires_grad=True)
    bias = torch.randn(channels, device="cuda", requires_grad=True)
    grad = torch.randn(shape, device="cuda").contiguous(memory_format=torch.channels_last)

    actual = kernels.get("group_norm_gelu")(x, weight, bias, 8, 1e-5)
    actual.backward(grad)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()
    expected = kernels.reference("group_norm_gelu")(
        x_ref, weight_ref, bias_ref, 8, 1e-5
    )
    expected.backward(grad)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(bias.grad, bias_ref.grad, atol=1e-3, rtol=1e-3)


def test_encoder_layout_and_parameter_names():
    encoder = FrameEncoder(d_model=512, image_size=64, stem_channels=32)
    parameter_names = set(dict(encoder.named_parameters()))
    assert "block2.b2_gn.weight" in parameter_names
    assert "block2.b2_gn.bias" in parameter_names
    assert "block3.b3_gn.weight" in parameter_names
    assert "block3.b3_gn.bias" in parameter_names
    for module in encoder.modules():
        if isinstance(module, torch.nn.Conv2d):
            assert module.weight.is_contiguous(memory_format=torch.channels_last)

    images = torch.randint(0, 256, (2, 5, 3, 64, 64), dtype=torch.uint8)
    prepared = prepare_images(images, 64)
    flat = prepared.flatten(0, 1)
    assert flat.is_contiguous(memory_format=torch.channels_last)
