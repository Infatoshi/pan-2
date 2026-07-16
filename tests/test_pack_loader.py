"""Pack-source loader unit tests (CPU): decode-window parity and _pack_items."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pan2.data.gpu_pipeline import (
    PipelineConfig,
    PipelinedGpuPretrainLoader,
    _decode_mp4_window,
)

ffmpeg_missing = pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg not installed",
)


def _mk_clip(path: Path, n_frames: int, size: int = 64) -> None:
    r = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i",
            f"testsrc2=size={size}x{size}:rate=10:duration={n_frames / 10}",
            "-frames:v", str(n_frames),
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "10", "-an",
            str(path),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr[-300:]


def _decode_sequential(mp4: Path, n_frames: int, size: int = 64) -> np.ndarray:
    """Reference: decode the first n_frames start-to-finish, no seek."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-frames:v", str(n_frames),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True,
    )
    assert r.returncode == 0
    need = n_frames * size * size * 3
    assert len(r.stdout) >= need, f"short decode {len(r.stdout)} < {need}"
    return np.frombuffer(r.stdout[:need], dtype=np.uint8).reshape(
        n_frames, size, size, 3
    )


@ffmpeg_missing
def test_decode_window_seek_is_frame_exact(tmp_path):
    """-ss before -i must land on the exact frame when decoding (GOP-10 clip)."""
    clip = tmp_path / "c.mkv"
    n = 120
    _mk_clip(clip, n)
    full = _decode_sequential(clip, n)
    for start in (0, 1, 9, 10, 11, 37, 100):  # GOP boundary straddles
        w = _decode_mp4_window(
            clip, start_frame=start, num_frames=20, image_size=64,
            fps=10.0, scale=False,
        )
        assert w.shape == (20, 64, 64, 3)
        assert np.array_equal(w, full[start : start + 20]), (
            f"seek mismatch at start={start}: "
            f"maxdiff={np.abs(w.astype(int) - full[start:start+20].astype(int)).max()}"
        )


@ffmpeg_missing
def test_decode_window_scale_flag_matches_reference(tmp_path):
    """scale=True path unchanged vs explicit reference scale command."""
    clip = tmp_path / "c128.mkv"
    _mk_clip(clip, 60, size=128)
    ref = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "1.000", "-i", str(clip),
         "-frames:v", "20", "-vf", "scale=64:64",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True,
    )
    assert ref.returncode == 0
    expected = np.frombuffer(ref.stdout, dtype=np.uint8).reshape(20, 64, 64, 3)
    got = _decode_mp4_window(
        clip, start_frame=10, num_frames=20, image_size=64, fps=10.0, scale=True
    )
    assert np.array_equal(got, expected)


def _make_index(tmp_path: Path) -> Path:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_pack_index as bpi

    ref64 = tmp_path / "ref64"
    ref64.mkdir(exist_ok=True)
    _mk_clip(ref64 / "vidAAAA111.mkv", 300)
    _mk_clip(ref64 / "vidBBBB222.mkv", 120)
    _mk_clip(ref64 / "vidCCCC333.mkv", 10)  # below t_load, must be skipped
    meta = tmp_path / "list.tsv"
    meta.write_text(
        "vidAAAA111\\tEthosLab\\t30\\tMinecraft longplay\\thttps://x/a\n"
        "vidBBBB222\\tEthosLab\\t12\\tDon't Starve\\thttps://x/b\n"
        "vidCCCC333\\tEthosLab\\t1\\tMinecraft short\\thttps://x/c\n"
    )
    out = tmp_path / "pack_index.npz"
    bpi.build(ref64, out, meta, workers=2)
    return out


def _loader_bare(cfg: PipelineConfig) -> PipelinedGpuPretrainLoader:
    """Construct without __init__ (CUDA ring) to unit-test item discovery."""
    obj = object.__new__(PipelinedGpuPretrainLoader)
    obj.cfg = cfg
    return obj


@ffmpeg_missing
def test_pack_items_max_start_and_filters(tmp_path):
    idx = _make_index(tmp_path)
    # t_load = 64 + 2*100 = 264 frames -> vidAAAA111 (300) only below both filters off
    cfg = PipelineConfig(
        prefer_source="pack",
        pack_index=str(idx),
        native_fps=10.0,
        image_size=64,
        context_len=64,
        min_goal_horizon=10,
        max_goal_horizon=100,
    )
    items = _loader_bare(cfg)._pack_items()
    assert [it["stem"] for it in items] == ["vidAAAA111"]
    assert items[0]["max_start"] == 300 - 264
    assert items[0]["native_size"] == 64

    # smaller t_load: both full episodes in; minecraft_only drops the Don't Starve one
    cfg2 = PipelineConfig(
        prefer_source="pack",
        pack_index=str(idx),
        native_fps=10.0,
        image_size=64,
        context_len=8,
        min_goal_horizon=5,
        max_goal_horizon=10,
        pack_minecraft_only=True,
    )
    items2 = _loader_bare(cfg2)._pack_items()
    assert [it["stem"] for it in items2] == ["vidAAAA111"]


@ffmpeg_missing
def test_pack_items_config_mismatch_raises(tmp_path):
    idx = _make_index(tmp_path)
    bad_fps = PipelineConfig(
        prefer_source="pack", pack_index=str(idx), native_fps=20.0, image_size=64
    )
    with pytest.raises(ValueError, match="fps"):
        _loader_bare(bad_fps)._pack_items()
    bad_size = PipelineConfig(
        prefer_source="pack", pack_index=str(idx), native_fps=10.0, image_size=96
    )
    with pytest.raises(ValueError, match="px"):
        _loader_bare(bad_size)._pack_items()


def test_pack_items_missing_index():
    cfg = PipelineConfig(prefer_source="pack", pack_index="/nonexistent/x.npz")
    with pytest.raises(FileNotFoundError):
        _loader_bare(cfg)._pack_items()
