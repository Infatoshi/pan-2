from __future__ import annotations

import torch
from torch.utils.data import Dataset


def synthetic_batch(
    batch_size: int,
    context_len: int,
    image_size: int,
    action_chunk: int,
    n_discrete: int,
    mouse_dim: int = 2,
    device: str | torch.device = "cpu",
    uint8: bool = False,
) -> dict[str, torch.Tensor]:
    if uint8:
        shape = (batch_size, context_len, 3, image_size, image_size)
        frames = torch.randint(0, 256, shape, device=device, dtype=torch.uint8)
        # future-frame goal (NOT a context frame), matching real-data task shape
        goal = torch.randint(
            0, 256, (batch_size, 3, image_size, image_size), device=device, dtype=torch.uint8
        )
        neg = torch.randint(
            0, 256, (batch_size, 3, image_size, image_size), device=device, dtype=torch.uint8
        )
    else:
        frames = torch.rand(batch_size, context_len, 3, image_size, image_size, device=device)
        future = torch.rand(batch_size, 3, image_size, image_size, device=device)
        goal = (future + 0.05 * torch.rand_like(future)).clamp(0, 1)
        neg = torch.rand(batch_size, 3, image_size, image_size, device=device)
    discrete = torch.randint(0, 2, (batch_size, action_chunk, n_discrete), device=device).float()
    mouse = torch.randn(batch_size, action_chunk, mouse_dim, device=device) * 0.1
    return {
        "frames": frames,
        "goal": goal,
        "neg": neg,
        "discrete": discrete,
        "mouse": mouse,
    }


class SyntheticGoalDataset(Dataset):
    def __init__(
        self,
        length: int = 256,
        context_len: int = 32,
        image_size: int = 64,
        action_chunk: int = 10,
        n_discrete: int = 23,
        uint8: bool = True,
    ):
        self.length = length
        self.context_len = context_len
        self.image_size = image_size
        self.action_chunk = action_chunk
        self.n_discrete = n_discrete
        self.uint8 = uint8

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        batch = synthetic_batch(
            1,
            self.context_len,
            self.image_size,
            self.action_chunk,
            self.n_discrete,
            uint8=self.uint8,
        )
        return {k: v.squeeze(0) for k, v in batch.items()}
