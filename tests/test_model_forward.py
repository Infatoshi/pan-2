import torch

from pan2.config import ModelConfig
from pan2.models.policy import PanPolicy


def test_forward_shapes_float():
    cfg = ModelConfig(
        image_size=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        context_len=8,
        action_chunk=4,
        n_discrete=23,
        frame_subsample=2,
        stem_channels=16,
    )
    m = PanPolicy(cfg)
    b = 2
    frames = torch.rand(b, cfg.context_len, 3, cfg.image_size, cfg.image_size)
    goal = torch.rand(b, 3, cfg.image_size, cfg.image_size)
    out = m(frames, goal, return_actions=True)
    assert out["contrastive_logits"].shape == (b, b)
    assert out["discrete_logits"].shape == (b, cfg.action_chunk, cfg.n_discrete)
    assert out["mouse_pred"].shape == (b, cfg.action_chunk, cfg.mouse_dim)
    # subsample: ceil pattern keeps last -> for T=8 k=2 expect 4 or 5 tokens
    assert out["frame_tok"].shape[0] == b
    assert out["frame_tok"].shape[1] <= cfg.context_len


def test_forward_uint8():
    cfg = ModelConfig(
        image_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        context_len=8,
        action_chunk=2,
        n_discrete=23,
        frame_subsample=4,
        stem_channels=16,
    )
    m = PanPolicy(cfg)
    b = 2
    frames = torch.randint(0, 256, (b, cfg.context_len, 3, 64, 64), dtype=torch.uint8)
    goal = torch.randint(0, 256, (b, 3, 64, 64), dtype=torch.uint8)
    out = m(frames, goal, return_actions=False)
    assert out["contrastive_logits"].shape == (b, b)


def test_forward_with_hard_negative():
    from pan2.train.losses import contrastive_loss

    cfg = ModelConfig(
        image_size=64, d_model=64, n_layers=2, n_heads=4,
        context_len=8, action_chunk=2, n_discrete=23,
        frame_subsample=2, stem_channels=16,
    )
    m = PanPolicy(cfg)
    b = 4
    frames = torch.rand(b, cfg.context_len, 3, 64, 64)
    goal = torch.rand(b, 3, 64, 64)
    neg = torch.rand(b, 3, 64, 64)
    out = m(frames, goal, neg)
    assert out["contrastive_logits"].shape == (b, b + 1)
    loss = contrastive_loss(out["contrastive_logits"])
    loss.backward()
    assert loss.item() == loss.item()


def test_contrastive_state_side_is_goal_blind():
    """Regression for the goal-identity leak: the state side of the
    contrastive logits must come from the last CONTEXT position, which is
    causally blind to the goal token. Swapping the goal must therefore leave
    each row's state embedding unchanged: logits row i under goal set A and
    goal set B must differ only through the goal projections, i.e.
    logits_A @ anything-goal-side changes but the state vectors match.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(
        image_size=64, d_model=64, n_layers=2, n_heads=4,
        context_len=8, action_chunk=2, n_discrete=23,
        frame_subsample=2, stem_channels=16,
    )
    m = PanPolicy(cfg).eval()
    b = 3
    frames = torch.rand(b, cfg.context_len, 3, 64, 64)
    goal_a = torch.rand(b, 3, 64, 64)
    goal_b = torch.rand(b, 3, 64, 64)
    with torch.no_grad():
        out_a = m(frames, goal_a)
        out_b = m(frames, goal_b)
        s_a = m.value_head.encode_state(out_a["state"])
        s_b = m.value_head.encode_state(out_b["state"])
    # same context, different goal -> identical state embeddings (bitwise:
    # same ops, same inputs on the context path)
    assert torch.equal(out_a["state"], out_b["state"])
    assert torch.equal(s_a, s_b)
    # and the logits must actually be built from that goal-blind state:
    # recompute from parts and compare against the model output
    with torch.no_grad():
        g_a = m.value_head.encode_goal(out_a["goal_tok"])
        expect = (s_a @ g_a.T) / 0.07
    assert torch.allclose(out_a["contrastive_logits"], expect, atol=0, rtol=0)
