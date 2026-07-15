from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


def _dw_block(name: str, cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                (
                    f"{name}_dw",
                    nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
                ),
                (f"{name}_pw", nn.Conv2d(cin, cout, 1, bias=False)),
                (f"{name}_gn", nn.GroupNorm(num_groups=min(8, cout), num_channels=cout)),
                (f"{name}_act", nn.GELU()),
            ]
        )
    )


class FrameEncoder(nn.Module):
    """Lightweight CNN: one token per frame.

    Named submodules so profilers can attribute conv/GN fwd+bwd per stage.
    Stages (64x64 in):
      stem  : Conv7s2 + GELU                         -> 32x32 (no GN)
      block1: Conv3s2 + GELU                         -> 16x16 (no DW, no GN)
      block2: DW+PW + GroupNorm + GELU               -> 8x8
      block3: DW+PW + GroupNorm + GELU               -> 4x4
      pool  : AdaptiveAvgPool2d(1)                   -> 1x1
    """

    def __init__(
        self,
        d_model: int = 512,
        in_channels: int = 3,
        image_size: int = 64,
        stem_channels: int = 32,
    ):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {image_size}")
        self.image_size = image_size
        c1 = stem_channels
        c2 = stem_channels * 2
        c3 = stem_channels * 4

        self.stem = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(in_channels, c1, 7, stride=2, padding=3, bias=False)),
                    ("act", nn.GELU()),
                ]
            )
        )
        # Softened early stage: single strided conv (no depthwise, no GN)
        self.block1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "b1_conv",
                        nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
                    ),
                    ("b1_act", nn.GELU()),
                ]
            )
        )
        self.block2 = _dw_block("b2", c2, c3, stride=2)  # 16->8
        self.block3 = _dw_block("b3", c3, d_model, stride=2)  # 8->4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # keep .net alias for anything that expects sequential
        self.net = nn.Sequential(self.stem, self.block1, self.block2, self.block3, self.pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.norm(self.proj(x))
