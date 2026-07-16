"""Checkpoint compat across torch.compile on/off builds.

kB wraps TransformerTemporal in torch.compile (OptimizedModule), which
nesting-changes state_dict keys (`temporal._orig_mod.*`). save/load must
round-trip between compiled and eager builds either way.
"""

from __future__ import annotations

import os

import pytest
import torch

from pan2.config import ModelConfig
from pan2.models.policy import PanPolicy
from pan2.train.loop import TrainState, load_ckpt, save_ckpt

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compile wrap requires CUDA"
)


def _mcfg() -> ModelConfig:
    return ModelConfig(
        image_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        context_len=16,
        action_chunk=4,
        n_discrete=5,
        mouse_dim=2,
        backbone="transformer",
        frame_subsample=1,
    )


def _build(compile_flag: bool, seed: int = 0) -> PanPolicy:
    os.environ["PAN2_TEMPORAL_COMPILE"] = "1" if compile_flag else "0"
    torch.manual_seed(seed)
    model = PanPolicy(_mcfg()).to("cuda")
    if compile_flag:
        assert hasattr(model.temporal, "_orig_mod"), "temporal not compiled"
    return model


def _state(model: torch.nn.Module) -> TrainState:
    return TrainState(
        model=model,
        optim=torch.optim.AdamW(model.parameters(), lr=1e-4),
        device=torch.device("cuda"),
    )


def _first_param_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    pa = next(a.parameters())
    pb = next(b.parameters())
    return torch.equal(pa, pb)


def test_ckpt_roundtrip_compiled_to_compiled(tmp_path):
    m1 = _build(compile_flag=True)
    p = tmp_path / "c.pt"
    save_ckpt(_state(m1), p)
    sd = torch.load(p, map_location="cpu", weights_only=True)["model"]
    assert not any("_orig_mod" in k for k in sd), "ckpt keys must stay plain"

    m2 = _build(compile_flag=True, seed=1)  # different init
    assert not _first_param_equal(m1, m2)
    load_ckpt(_state(m2), p)
    assert _first_param_equal(m1, m2)


def test_ckpt_cross_directions(tmp_path):
    # compiled build writes; eager build reads
    mc = _build(compile_flag=True)
    p1 = tmp_path / "compiled.pt"
    save_ckpt(_state(mc), p1)
    me = _build(compile_flag=False, seed=1)
    load_ckpt(_state(me), p1)
    assert _first_param_equal(mc, me)

    # eager build writes; compiled build reads
    p2 = tmp_path / "eager.pt"
    save_ckpt(_state(me), p2)
    mc2 = _build(compile_flag=True, seed=2)
    load_ckpt(_state(mc2), p2)
    assert _first_param_equal(me, mc2)
