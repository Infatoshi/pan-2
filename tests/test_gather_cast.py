"""gather_cast / scale_cast unit tests: optimized vs pure-torch reference.

Value semantics are uint8 -> fp32 mul -> single rounding at out dtype, so
comparison is bitwise (ints exact, floats bit-identical), not atol/rtol.
"""

from __future__ import annotations

import pytest
import torch

from pan2 import kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")

SCALE = 1.0 / 255.0


def _u8(*shape, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g).cuda()


@pytest.mark.parametrize("shape", [(2080, 3, 64, 64), (64, 3, 64, 64), (7, 3, 13, 11)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_scale_cast_bitwise(shape, out_dtype):
    x = _u8(*shape)
    want = kernels.reference("scale_cast")(x, SCALE, out_dtype)
    got = kernels.get("scale_cast")(x, SCALE, out_dtype)
    assert got.dtype == out_dtype
    assert got.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(want, got)


def test_scale_cast_5d():
    x = _u8(4, 65, 3, 16, 16)
    want = kernels.reference("scale_cast")(x, SCALE, torch.bfloat16)
    got = kernels.get("scale_cast")(x, SCALE, torch.bfloat16)
    assert got.shape == x.shape
    assert torch.equal(want, got)


def test_gather_cast_bitwise_and_packing():
    s, t, t_ctx, b = 8, 7, 5, 4
    ring = _u8(s, t, 3, 64, 64, seed=1)
    slots = torch.tensor([0, 0, 5, 3], device="cuda", dtype=torch.long)
    want_ctx, want_tail = kernels.reference("gather_cast")(
        ring, slots, SCALE, torch.bfloat16, t_ctx
    )
    got_ctx, got_tail = kernels.get("gather_cast")(
        ring, slots, SCALE, torch.bfloat16, t_ctx
    )
    assert got_ctx.shape == (b * t_ctx, 3, 64, 64)
    assert got_tail.shape == (b * (t - t_ctx), 3, 64, 64)
    assert torch.equal(want_ctx, got_ctx)
    assert torch.equal(want_tail, got_tail)
    # tail row order: per-batch [goal, neg] as baked in the slot layout
    tail4 = got_tail.view(b, 2, 3, 64, 64)
    ref = ring.index_select(0, slots)
    for i in range(b):
        expect_goal = ref[i, t_ctx].to(torch.float32).mul_(SCALE).to(torch.bfloat16)
        expect_neg = ref[i, t_ctx + 1].to(torch.float32).mul_(SCALE).to(torch.bfloat16)
        assert torch.equal(tail4[i, 0].to(torch.bfloat16), expect_goal)
        assert torch.equal(tail4[i, 1].to(torch.bfloat16), expect_neg)


def test_gather_cast_production_shape():
    s, t, t_ctx, b = 64, 67, 65, 32
    ring = _u8(s, t, 3, 64, 64, seed=2)
    g = torch.Generator(device="cpu").manual_seed(3)
    slots = torch.randint(0, s, (b,), generator=g).cuda()
    want = kernels.reference("gather_cast")(ring, slots, SCALE, torch.bfloat16, t_ctx)
    got = kernels.get("gather_cast")(ring, slots, SCALE, torch.bfloat16, t_ctx)
    assert torch.equal(want[0], got[0])
    assert torch.equal(want[1], got[1])


def test_fallback_non_contiguous_uses_ref_values():
    s, t, t_ctx = 4, 6, 4
    base = _u8(s, t, 3, 64, 128, seed=4)
    ring = base[..., ::2]  # non-contiguous view -> ref path
    assert not ring.is_contiguous()
    slots = torch.tensor([1, 3], device="cuda", dtype=torch.long)
    want = kernels.reference("gather_cast")(ring, slots, SCALE, torch.bfloat16, t_ctx)
    got = kernels.get("gather_cast")(ring, slots, SCALE, torch.bfloat16, t_ctx)
    assert torch.equal(want[0], got[0])
    assert torch.equal(want[1], got[1])


def test_cpu_scale_cast_matches_ref():
    x = torch.randint(0, 256, (5, 3, 13, 11), dtype=torch.uint8)
    want = kernels.reference("scale_cast")(x, SCALE, torch.float32)
    got = kernels.get("scale_cast")(x, SCALE, torch.float32)
    assert torch.equal(want, got)


def test_prepare_images_uses_fused_cast():
    """prepare_images on uint8 CUDA equals the old fp32 chain, bit-exact."""
    from pan2.models.preprocess import prepare_images

    x = _u8(2, 65, 3, 64, 64, seed=5)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = prepare_images(x, 64)
    assert got.dtype == torch.bfloat16
    want_ctx = (
        x.reshape(-1, 3, 64, 64)[..., :]
        .to(torch.float32)
        .mul_(SCALE)
        .to(torch.bfloat16)
        .reshape(2, 65, 3, 64, 64)
    )
    assert torch.equal(got.reshape(want_ctx.shape), want_ctx)


def test_preprocess_no_autocast_is_fp32():
    from pan2.models.preprocess import prepare_images

    x = _u8(2, 3, 64, 64, seed=6)
    got = prepare_images(x, 64)
    assert got.dtype == torch.float32
