import pytest
import torch

from pan2 import kernels
from pan2.kernels import conv_gelu as conv_gelu_mod
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
@pytest.mark.skipif(not conv_gelu_mod._HAS_TRITON, reason="Triton not installed")
@pytest.mark.parametrize(
    "input_shape,weight_shape,padding",
    [
        ((64, 3, 64, 64), (32, 3, 7, 7), 3),
        ((64, 32, 32, 32), (64, 32, 3, 3), 1),
        ((2080, 32, 32, 32), (64, 32, 3, 3), 1),
    ],
)
def test_conv_gelu_triton_path_matches_reference(
    monkeypatch, input_shape, weight_shape, padding
):
    """Force Triton ON and assert we are not comparing ref-vs-ref."""
    monkeypatch.setenv("PAN2_CONV_GELU_TRITON", "1")
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
    assert conv_gelu_mod._can_use_triton(x, weight, 2, padding)

    actual = kernels.get("conv_gelu")(x, weight, 2, padding)
    grad = (
        torch.randn_like(actual)
        .mul_(1e-5)
        .contiguous(memory_format=torch.channels_last)
    )
    actual.backward(grad)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    expected = conv_gelu_mod.conv_gelu_ref(x_ref, weight_ref, 2, padding)
    expected.backward(grad)

    assert torch.isfinite(actual).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(weight.grad).all()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
@pytest.mark.skipif(not conv_gelu_mod._HAS_TRITON, reason="Triton not installed")
def test_dgrad_immune_to_nan_past_weight_buffer():
    """Regression: generic dgrad must not load weight at kh>=KH / kw>=KW.

    The old parity-strided loop used trip count ceil(K/stride), so with
    kh_start=1, KH=3 it issued kh=3. Masked dpre (0) times OOB weight nan
    poisons tl.dot. Place nan immediately after a packed channels-last
    weight buffer and require finite dx.
    """
    import triton

    N, CIN, H, W = 4, 32, 32, 32
    COUT, KH, KW = 64, 3, 3
    STRIDE, PADDING = 2, 1
    OH = (H + 2 * PADDING - KH) // STRIDE + 1
    OW = (W + 2 * PADDING - KW) // STRIDE + 1
    ke = KH * KW * CIN
    packed = COUT * ke

    torch.manual_seed(0)
    storage = torch.empty(packed + 8192, device="cuda", dtype=torch.bfloat16)
    storage[:packed] = torch.randn(packed, device="cuda", dtype=torch.bfloat16)
    storage[packed:] = float("nan")
    weight = torch.as_strided(
        storage[:packed],
        size=(COUT, CIN, KH, KW),
        stride=(ke, 1, KW * CIN, CIN),
    )
    assert weight.is_contiguous(memory_format=torch.channels_last)
    assert torch.isfinite(weight).all()

    dpre = (
        torch.randn(N, COUT, OH, OW, device="cuda", dtype=torch.bfloat16)
        .contiguous(memory_format=torch.channels_last)
    )
    dx = (
        torch.zeros(N, CIN, H, W, device="cuda", dtype=torch.bfloat16)
        .contiguous(memory_format=torch.channels_last)
    )
    block_m = 256
    half = H // STRIDE
    grid = (triton.cdiv(N * half * half, block_m), STRIDE * STRIDE)
    conv_gelu_mod._conv_gelu_dgrad_kernel[grid](
        dpre,
        weight,
        dx,
        N,
        CIN=CIN,
        COUT=COUT,
        H=H,
        W=W,
        KH=KH,
        KW=KW,
        OH=OH,
        OW=OW,
        STRIDE=STRIDE,
        PADDING=PADDING,
        BLOCK_M=block_m,
        BLOCK_CIN=32,
        BLOCK_COUT=32,
        num_warps=8,
        num_stages=2,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(dx).all(), "dgrad produced nan from finite inputs + nan tail"


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
