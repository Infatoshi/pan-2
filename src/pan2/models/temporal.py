from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pan2.kernels  # noqa: F401  — ensure kernel modules register
from pan2.kernels import get


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no")


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention via SDPA (FlashAttention when available)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        b, length, d = x.shape
        # reshape then unbind avoids SelectBackward materializing full copies
        qkv = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, Hd]
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        # reshape (not view) handles the transpose without an extra .contiguous()
        y = y.transpose(1, 2).reshape(b, length, d)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        # Keep parameter names fc1.weight / fc1.bias (state_dict stable).
        # Forward uses weight-only linear + fused bias_gelu to cut a memory pass.
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_gelu = get("bias_gelu")
        # weight gemm without bias, then fused bias+GELU (one elementwise pass)
        h = F.linear(x, self.fc1.weight)
        h = bias_gelu(h, self.fc1.bias)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self._d_model = d_model
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_add = get("residual_add")
        layer_norm = get("layer_norm_affine")
        shape = (self._d_model,)
        h = layer_norm(x, shape, self.norm1.weight, self.norm1.bias, self.norm1.eps)
        x = residual_add(x, self.attn(h))
        h = layer_norm(x, shape, self.norm2.weight, self.norm2.bias, self.norm2.eps)
        x = residual_add(x, self.mlp(h))
        return x


class TransformerTemporal(nn.Module):
    """Causal transformer over frame tokens + goal token (SDPA/Flash path)."""

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_len: int = 512,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self._d_model = d_model
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        layer_norm = get("layer_norm_affine")
        _, length, _ = tokens.shape
        if length > self.pos.shape[1]:
            raise ValueError(f"sequence length {length} > max_len {self.pos.shape[1]}")
        # Broadcast pos add (not same-storage residual); keep plain + for grad to pos.
        x = tokens + self.pos[:, :length]
        for block in self.blocks:
            x = block(x)
        return layer_norm(
            x,
            (self._d_model,),
            self.norm.weight,
            self.norm.bias,
            self.norm.eps,
        )


class IdentityTemporal(nn.Module):
    def __init__(self, d_model: int = 512, **_: object):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self._d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        layer_norm = get("layer_norm_affine")
        return layer_norm(
            tokens,
            (self._d_model,),
            self.norm.weight,
            self.norm.bias,
            self.norm.eps,
        )


def build_temporal(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("transformer", "sdpa"):
        module: nn.Module = TransformerTemporal(**kwargs)
        # Scoped torch.compile on the temporal stack: fuses residual/elementwise
        # copies that Triton alone cannot eliminate across attention boundaries.
        # Disable with PAN2_TEMPORAL_COMPILE=0 for debugging.
        # PAN2_TEMPORAL_COMPILE_MODE selects the inductor mode. Keep "default"
        # in production: cudagraph trees ("reduce-overhead") recover only
        # ~0.34 ms/step of launch gaps (7.19 -> 6.85 ms wall, GPU0 2026-07-15).
        # The NaNs once blamed on them (2026-07-15 hunt) were the conv_gelu
        # dgrad flake, root-caused and fixed same-day (kF; DEVLOG) - "ro" mode
        # was never causal, just not worth 0.34 ms of graph-capture risk.
        if (
            _env_flag("PAN2_TEMPORAL_COMPILE", default=True)
            and torch.cuda.is_available()
        ):
            mode = os.environ.get("PAN2_TEMPORAL_COMPILE_MODE", "default")
            module = torch.compile(module, mode=mode, fullgraph=False)
        return module
    if name == "identity":
        return IdentityTemporal(**kwargs)
    raise ValueError(f"unknown temporal backbone: {name}")
