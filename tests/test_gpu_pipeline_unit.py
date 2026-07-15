"""CPU-side unit tests for pipeline helpers (no GPU required)."""

from pan2.data.gpu_pipeline import _subsample_indices


def test_subsample_keeps_last():
    idxs = _subsample_indices(128, 4)
    assert idxs[0] == 0
    assert idxs[-1] == 127
    assert len(idxs) == 33  # 0,4,...,124 + 127


def test_subsample_aligned():
    idxs = _subsample_indices(8, 2)
    assert idxs[-1] == 7
    assert 0 in idxs
