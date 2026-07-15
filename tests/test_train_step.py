from pan2.config import Config, ModelConfig, TrainConfig
from pan2.data.synthetic import synthetic_batch
from pan2.train.loop import build_state, train_steps


def test_pretrain_step_finite():
    cfg = Config(
        model=ModelConfig(
            d_model=64, n_layers=2, n_heads=4, context_len=8, action_chunk=4, image_size=64
        ),
        train=TrainConfig(stage="pretrain", batch_size=4, bf16=False, device="cpu", max_steps=2),
    )
    state = build_state(cfg)

    def gen():
        while True:
            yield synthetic_batch(4, 8, 64, 4, 23, uint8=True)

    logs = train_steps(state, cfg, gen(), n_steps=2)
    assert logs[-1]["loss"] == logs[-1]["loss"]
    assert state.step == 2
