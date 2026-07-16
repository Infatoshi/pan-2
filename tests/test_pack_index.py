"""Pack index builder tests on synthetic ffmpeg fixtures."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

sys_path_added = False


def _ensure_import():
    global sys_path_added
    if not sys_path_added:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        sys_path_added = True


ffmpeg_missing = pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg not installed",
)


def _mk_clip(path: Path, n_frames: int, size: int = 64) -> None:
    """Deterministic x264 GOP-10 clip at 10fps (test content, CBC-free)."""
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


@ffmpeg_missing
def test_build_pack_index_fields_and_filters(tmp_path):
    _ensure_import()
    import build_pack_index as bpi

    ref64 = tmp_path / "ref64"
    ref64.mkdir()
    _mk_clip(ref64 / "vidAAAA111.mkv", 40)
    _mk_clip(ref64 / "vidBBBB222.mkv", 90)

    meta = tmp_path / "list.tsv"
    meta.write_text(
        "vidAAAA111\\tEthosLab\\t1200\\tMinecraft - HermitCraft #1\\thttps://x/y1\n"
        "vidBBBB222\\tEthosLab\\t900\\tDon't Starve Together #38\\thttps://x/y2\n"
        # duplicate meta line for the first id must be ignored
        "vidAAAA111\\tEthosLab\\t1200\\tMinecraft - HermitCraft #1\\thttps://x/y1\n"
    )

    out = tmp_path / "pack_index.npz"
    header = bpi.build(ref64, out, meta, workers=2)

    assert header["episodes"] == 2
    assert header["total_frames"] == 130
    assert header["dropped_probe_failures"] == 0
    assert header["minecraft_fraction"] == pytest.approx(0.5)

    z = np.load(out)
    assert int(z["version"]) == 1
    assert float(z["fps"]) == pytest.approx(10.0)
    assert int(z["image_size"]) == 64
    assert int(z["gop"]) == 20

    order = list(z["stem"])
    assert order == ["vidAAAA111", "vidBBBB222"]
    assert list(z["n_frames"]) == [40, 90]
    assert list(z["minecraft"]) == [True, False]
    assert list(z["channel"]) == ["EthosLab", "EthosLab"]
    assert z["duration_s"][1] == pytest.approx(9.0)
    for p in z["path"]:
        assert Path(str(p)).is_file()


@ffmpeg_missing
def test_build_pack_index_drops_corrupt(tmp_path):
    _ensure_import()
    import build_pack_index as bpi

    ref64 = tmp_path / "ref64"
    ref64.mkdir()
    _mk_clip(ref64 / "goodAAAAAA1.mkv", 50)
    (ref64 / "deadBBBBBB2.mkv").write_bytes(b"not a video")

    out = tmp_path / "pack_index.npz"
    header = bpi.build(ref64, out, tmp_path / "no-meta.tsv", workers=2)
    assert header["episodes"] == 1
    assert header["dropped_probe_failures"] == 1
    assert header["meta_coverage"] == pytest.approx(0.0)
    z = np.load(out)
    assert list(z["stem"]) == ["goodAAAAAA1"]
