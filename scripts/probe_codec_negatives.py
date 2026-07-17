#!/usr/bin/env python3
"""Does crf28@64px keep wrong-horizon negatives distinguishable?

For stems present in both the crawl raw/ and ref64/ trees, decode the same
initial window two ways:
  clean : raw mp4 -> fps=10 -> scale 64x64 lanczos   (ideal training view)
  pack  : ref64 mkv decoded as-is                    (what training reads)

Reports, in pixel RMS units (uint8 0..255):
  codec_err        rms(pack[t] - clean[t])           full-pipeline codec error
  sep(h)           rms(clean[t] - clean[t+h])        temporal signal at horizon h
  ratio(h)         codec_err / sep(h)

A wrong-horizon negative at h is discriminable when the codec error is well
below the temporal separation (ratio << 1). Horizons in native 10fps frames;
h=10 is the min goal horizon (1s).

Usage:
  uv run python scripts/probe_codec_negatives.py \
      --raw-dir data/crawl/raw --ref64-dir data/crawl/ref64 \
      --sample-n 12 --frames 240 --workers 4
"""
from __future__ import annotations

import argparse
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HORIZONS = (1, 5, 10, 50, 150)


def _decode(cmd: list[str], n: int, size: int) -> np.ndarray:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    need = n * size * size * 3
    if len(p.stdout) < need:
        raise RuntimeError(f"short decode {cmd[-2]} got={len(p.stdout)} need={need}")
    return (
        np.frombuffer(p.stdout[:need], dtype=np.uint8)
        .reshape(n, size, size, 3)
        .astype(np.float32)
    )


def decode_clean(raw: Path, n: int, size: int) -> np.ndarray:
    return _decode(
        ["ffmpeg", "-v", "error", "-i", str(raw),
         "-vf", f"fps=10,scale={size}:{size}:flags=lanczos",
         "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        n, size,
    )


def decode_pack(ref64: Path, n: int, size: int) -> np.ndarray:
    return _decode(
        ["ffmpeg", "-v", "error", "-i", str(ref64),
         "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        n, size,
    )


def probe_stem(raw: Path, ref64: Path, n: int, size: int) -> dict[str, float]:
    clean = decode_clean(raw, n, size)
    pack = decode_pack(ref64, n, size)
    out = {"codec_err": float(np.sqrt(np.mean((pack - clean) ** 2)))}
    for h in HORIZONS:
        if h >= n:
            continue
        d = clean[:-h] - clean[h:]
        out[f"sep_{h}"] = float(np.sqrt(np.mean(d**2)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/crawl/raw")
    p.add_argument("--ref64-dir", default="data/crawl/ref64")
    p.add_argument("--sample-n", type=int, default=12)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ref64_dir = Path(args.ref64_dir)
    raw_dir = Path(args.raw_dir)
    pairs = []
    for mkv in ref64_dir.iterdir():
        stem = mkv.stem
        for ext in (".mp4", ".webm", ".mkv"):
            raw = raw_dir / f"{stem}{ext}"
            if raw.exists():
                pairs.append((raw, mkv))
                break
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.sample_n]
    if not pairs:
        raise SystemExit("no raw/ref64 stem pairs found")
    print(f"probing {len(pairs)} stems, {args.frames} frames each")

    results: list[dict[str, float]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(probe_stem, raw, mkv, args.frames, args.image_size): mkv.stem
            for raw, mkv in pairs
        }
        for fut, stem in futs.items():
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001 - per-stem failures are data
                print(f"  {stem}: FAILED {e}")
                continue
            results.append(r)
            seps = " ".join(f"sep{h}={r.get(f'sep_{h}', float('nan')):.2f}" for h in HORIZONS)
            print(f"  {stem}: codec_err={r['codec_err']:.2f} {seps}")

    if not results:
        raise SystemExit("all probes failed")
    print("\n=== aggregate (mean over stems) ===")
    ce = float(np.mean([r["codec_err"] for r in results]))
    print(f"codec_err_rms={ce:.2f}  (uint8 units)")
    for h in HORIZONS:
        vals = [r[f"sep_{h}"] for r in results if f"sep_{h}" in r]
        if not vals:
            continue
        sep = float(np.mean(vals))
        print(f"h={h:4d}  sep_rms={sep:6.2f}  codec/sep={ce / sep:5.2f}")


if __name__ == "__main__":
    main()
