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
