"""Fused multi-tensor AdamW vs torch.optim.AdamW(fused=True)."""

from __future__ import annotations

import pytest
import torch

from pan2.config import ModelConfig
from pan2.kernels import get, reference
from pan2.kernels.fused_adamw import FusedAdamW
from pan2.models.policy import PanPolicy


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _production_cfg() -> ModelConfig:
    return ModelConfig(
        image_size=64,
        d_model=512,
        n_layers=8,
        n_heads=8,
        context_len=128,
        action_chunk=10,
        n_discrete=23,
        mouse_dim=2,
        backbone="transformer",
        frame_subsample=1,
    )


def _clone_params(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _set_params(dst: list[torch.nn.Parameter], src: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for d, s in zip(dst, src, strict=True):
            d.copy_(s)


def _fill_grads(params: list[torch.nn.Parameter], seed: int) -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for p in params:
        noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
        buf = torch.empty_like(p)
        buf.copy_(noise.to(device=p.device, dtype=p.dtype))
        p.grad = buf


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


@pytest.mark.parametrize("device", _devices())
def test_adamw_step_ref_matches_torch_single_tensor(device: str) -> None:
    """Pure-torch ref step matches single-tensor torch AdamW math."""
    torch.manual_seed(0)
    p = torch.randn(17, 13, device=device, dtype=torch.float32)
    g = torch.randn_like(p)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    step = torch.zeros((), dtype=torch.float32, device=device)

    p_t = p.clone().requires_grad_(True)
    p_t.grad = g.clone()
    opt = torch.optim.AdamW(
        [p_t], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, fused=False
    )
    # force state init
    opt.step()
    # reset to same start for fair compare of one step from zero state
    with torch.no_grad():
        p_t.copy_(p)
    opt.state[p_t]["exp_avg"].zero_()
    opt.state[p_t]["exp_avg_sq"].zero_()
    opt.state[p_t]["step"].zero_()
    p_t.grad = g.clone()
    opt.step()

    p_r, m_r, v_r, step_r = p.clone(), m.clone(), v.clone(), step.clone()
    reference("fused_adamw")(
        [p_r],
        [g.clone()],
        [m_r],
        [v_r],
        [step_r],
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
        eps=1e-8,
    )
    assert _max_abs(p_r, p_t.detach()) <= 1e-6
    assert _max_abs(m_r, opt.state[p_t]["exp_avg"]) <= 1e-5
    assert _max_abs(v_r, opt.state[p_t]["exp_avg_sq"]) <= 1e-5
    assert abs(float(step_r.item()) - float(opt.state[p_t]["step"].item())) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused path")
def test_adamw_opt_vs_ref_small_cuda() -> None:
    """Triton multi-tensor path matches pure-torch ref on mixed shapes."""
    device = "cuda"
    torch.manual_seed(1)
    shapes = [(128,), (64, 64), (8, 3, 3, 3), (7, 5)]
    params = [torch.randn(*s, device=device, dtype=torch.float32) for s in shapes]
    # channels_last dense 4-d
    params[2] = params[2].to(memory_format=torch.channels_last)
    grads = [torch.randn_like(p) for p in params]
    ms = [torch.zeros_like(p) for p in params]
    vs = [torch.zeros_like(p) for p in params]
    steps = [torch.zeros((), dtype=torch.float32, device=device) for _ in params]

    p_r = [p.clone() for p in params]
    g_r = [g.clone() for g in grads]
    m_r = [m.clone() for m in ms]
    v_r = [v.clone() for v in vs]
    s_r = [s.clone() for s in steps]

    p_o = [p.clone() for p in params]
    g_o = [g.clone() for g in grads]
    m_o = [m.clone() for m in ms]
    v_o = [v.clone() for v in vs]
    s_o = [s.clone() for s in steps]

    kwargs = dict(lr=3e-4, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8)
    for _ in range(5):
        reference("fused_adamw")(p_r, g_r, m_r, v_r, s_r, **kwargs)
        get("fused_adamw")(p_o, g_o, m_o, v_o, s_o, **kwargs)

    for a, b in zip(p_r, p_o, strict=True):
        assert _max_abs(a, b) <= 1e-6
    for a, b in zip(m_r, m_o, strict=True):
        assert _max_abs(a, b) <= 1e-5
    for a, b in zip(v_r, v_o, strict=True):
        assert _max_abs(a, b) <= 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA production equiv")
def test_200_step_equivalence_production_panpolicy() -> None:
    """200-step param/state match vs torch fused AdamW on production PanPolicy."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    cfg = _production_cfg()
    model_t = PanPolicy(cfg).to(device)
    model_o = PanPolicy(cfg).to(device)
    model_o.load_state_dict(model_t.state_dict())

    params_t = [p for p in model_t.parameters() if p.requires_grad]
    params_o = [p for p in model_o.parameters() if p.requires_grad]
    assert len(params_t) == len(params_o)

    opt_t = torch.optim.AdamW(
        params_t, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, fused=True
    )
    opt_o = FusedAdamW(
        params_o, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    n_steps = 200
    for step in range(n_steps):
        _fill_grads(params_t, seed=1000 + step)
        # identical grads on both
        for a, b in zip(params_t, params_o, strict=True):
            assert a.grad is not None
            b.grad = a.grad.detach().clone()
        opt_t.step()
        opt_o.step()

    max_p = 0.0
    max_m = 0.0
    max_v = 0.0
    for pt, po in zip(params_t, params_o, strict=True):
        max_p = max(max_p, _max_abs(pt, po))
        st = opt_t.state[pt]
        so = opt_o.state[po]
        max_m = max(max_m, _max_abs(st["exp_avg"], so["exp_avg"]))
        max_v = max(max_v, _max_abs(st["exp_avg_sq"], so["exp_avg_sq"]))
        assert abs(float(st["step"].item()) - float(so["step"].item())) < 1e-5

    assert max_p <= 1e-6, f"param max|diff|={max_p}"
    assert max_m <= 1e-5, f"exp_avg max|diff|={max_m}"
    assert max_v <= 1e-5, f"exp_avg_sq max|diff|={max_v}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA state_dict round-trip")
def test_state_dict_roundtrip_both_directions() -> None:
    """ours -> torch fused -> continue; torch fused -> ours -> continue."""
    device = torch.device("cuda")
    torch.manual_seed(2)
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, context_len=16)
    model = PanPolicy(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    def run_steps(opt: torch.optim.Optimizer, n: int, seed0: int) -> None:
        for i in range(n):
            _fill_grads(params, seed0 + i)
            opt.step()

    # --- ours -> torch ---
    model_a = PanPolicy(cfg).to(device)
    model_a.load_state_dict(model.state_dict())
    params_a = [p for p in model_a.parameters() if p.requires_grad]
    opt_ours = FusedAdamW(params_a, lr=1e-3, weight_decay=0.01)
    for i in range(5):
        _fill_grads(params_a, 50 + i)
        opt_ours.step()
    sd_ours = opt_ours.state_dict()
    # snapshot params after 5 ours steps
    snap = _clone_params(params_a)

    opt_torch = torch.optim.AdamW(params_a, lr=1e-3, weight_decay=0.01, fused=True)
    opt_torch.load_state_dict(sd_ours)
    for i in range(5):
        _fill_grads(params_a, 100 + i)
        opt_torch.step()
    after_torch = _clone_params(params_a)

    # pure ours 10 steps should match 5 ours + 5 torch
    model_b = PanPolicy(cfg).to(device)
    model_b.load_state_dict(model.state_dict())
    params_b = [p for p in model_b.parameters() if p.requires_grad]
    opt_b = FusedAdamW(params_b, lr=1e-3, weight_decay=0.01)
    for i in range(5):
        _fill_grads(params_b, 50 + i)
        opt_b.step()
    for i in range(5):
        _fill_grads(params_b, 100 + i)
        opt_b.step()
    for a, b in zip(after_torch, params_b, strict=True):
        assert _max_abs(a, b) <= 1e-5

    # --- torch -> ours ---
    model_c = PanPolicy(cfg).to(device)
    model_c.load_state_dict(model.state_dict())
    params_c = [p for p in model_c.parameters() if p.requires_grad]
    opt_t = torch.optim.AdamW(params_c, lr=1e-3, weight_decay=0.01, fused=True)
    for i in range(5):
        _fill_grads(params_c, 200 + i)
        opt_t.step()
    sd_t = opt_t.state_dict()

    opt_o2 = FusedAdamW(params_c, lr=1e-3, weight_decay=0.01)
    opt_o2.load_state_dict(sd_t)
    for i in range(5):
        _fill_grads(params_c, 300 + i)
        opt_o2.step()
    after_ours = _clone_params(params_c)

    model_d = PanPolicy(cfg).to(device)
    model_d.load_state_dict(model.state_dict())
    params_d = [p for p in model_d.parameters() if p.requires_grad]
    opt_d = torch.optim.AdamW(params_d, lr=1e-3, weight_decay=0.01, fused=True)
    for i in range(5):
        _fill_grads(params_d, 200 + i)
        opt_d.step()
    for i in range(5):
        _fill_grads(params_d, 300 + i)
        opt_d.step()
    for a, b in zip(after_ours, params_d, strict=True):
        assert _max_abs(a, b) <= 1e-5

    # unused snap just ensures we captured mid-state
    assert snap[0].shape == params_a[0].shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA build_state flag")
def test_build_state_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from pan2.config import Config, TrainConfig
    from pan2.kernels.fused_adamw import FusedAdamW
    from pan2.train.loop import build_state

    cfg = Config(
        model=ModelConfig(d_model=64, n_layers=2, n_heads=4, context_len=16),
        train=TrainConfig(device="cuda", compile=False, synthetic=True),
    )
    monkeypatch.setenv("PAN2_FUSED_ADAMW", "1")
    st = build_state(cfg)
    assert isinstance(st.optim, FusedAdamW)

    monkeypatch.setenv("PAN2_FUSED_ADAMW", "0")
    st2 = build_state(cfg)
    assert type(st2.optim) is torch.optim.AdamW


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA ckpt schema")
def test_train_state_ckpt_schema_compatible() -> None:
    """Optimizer state_dict keys match torch AdamW (save/load path)."""
    device = torch.device("cuda")
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, context_len=16)
    model = PanPolicy(cfg).to(device)
    params = list(model.parameters())
    ours = FusedAdamW(params, lr=3e-4, weight_decay=0.01)
    torch_opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01, fused=True)
    _fill_grads(params, 0)
    ours.step()
    # fresh model for torch
    model2 = PanPolicy(cfg).to(device)
    model2.load_state_dict(model.state_dict())
    params2 = list(model2.parameters())
    torch_opt = torch.optim.AdamW(params2, lr=3e-4, weight_decay=0.01, fused=True)
    _fill_grads(params2, 0)
    torch_opt.step()

    sd_o = ours.state_dict()
    sd_t = torch_opt.state_dict()
    assert set(sd_o.keys()) == set(sd_t.keys()) == {"state", "param_groups"}
    # per-param state keys
    st_o = next(iter(sd_o["state"].values()))
    st_t = next(iter(sd_t["state"].values()))
    assert set(st_o.keys()) == set(st_t.keys()) == {"step", "exp_avg", "exp_avg_sq"}
    assert st_o["step"].dtype == torch.float32
    assert st_o["step"].device.type == "cuda"
