from __future__ import annotations

import torch


def configure_cuda_fast_math() -> None:
    """Enable high-throughput CUDA defaults for training."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Prefer TF32 tensor cores on Ampere+ for matmul/convs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
