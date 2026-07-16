from __future__ import annotations

import torch
import torch.nn.functional as F

from pan2 import kernels


def _cast_dtype() -> torch.dtype:
    """Destination dtype for uint8 images: the autocast dtype when active."""
    if torch.cuda.is_available() and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return torch.float32


def prepare_images(
    x: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """uint8/float NCHW or BTCHW -> channels-last float in [0,1]."""
    unflatten: tuple[int, int] | None = None
    if x.ndim == 5:
        b, t, c, h, w = x.shape
        unflatten = (b, t)
        x = x.reshape(b * t, c, h, w)
    elif x.ndim == 4:
        _, _, h, w = x.shape
    else:
        raise ValueError(f"expected 4D or 5D image tensor, got {tuple(x.shape)}")

    if x.dtype == torch.uint8:
        x = kernels.get("scale_cast")(x, 1.0 / 255.0, _cast_dtype())
    elif x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        x = x.to(dtype=torch.float32, memory_format=torch.channels_last)
    else:
        x = x.contiguous(memory_format=torch.channels_last)

    if h != image_size or w != image_size:
        x = F.interpolate(
            x, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    if unflatten is not None:
        x = x.unflatten(0, unflatten)
    return x
