import torch

from pan2.models.preprocess import prepare_images


def test_uint8_normalize():
    x = torch.randint(0, 256, (2, 4, 3, 64, 64), dtype=torch.uint8)
    y = prepare_images(x, 64)
    assert y.dtype == torch.float32
    assert y.shape == x.shape
    assert float(y.max()) <= 1.0 + 1e-5


def test_gpu_resize():
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    y = prepare_images(x, 128)
    assert y.shape[-2:] == (128, 128)
