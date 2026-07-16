#!/usr/bin/env python3
"""PSNR/SSIM of codec variants against the clean training-view reference.

Reference per episode: raw mp4 -> fps=10 -> scale 128x128 lanczos (the exact
pixels any training-view encode starts from). Variants are compared after
decode, cropped to the top-left 128x128 (no-op for unpadded encodes; strips
the black pad used to satisfy NVENC min frame dims).

Usage:
  uv run python scripts/measure_codec_quality.py \
      --raw-dir /data/pan-2/raw \
      --variant-dirs /data/pan-2/encode_ablation/crf23 [...] \
      --sample-n 12 --out metrics.csv --workers 3
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PSNR_RE = re.compile(r"PSNR.*average:([\d.]+|inf)")
SSIM_RE = re.compile(r"SSIM.*All:([\d.]+)")


def measure(variant: Path, raw: Path, image_size: int) -> tuple[float, float]:
    """-> (psnr_db_avg, ssim_avg) of variant vs reference view of raw."""
    graph = (
        f"[0:v]crop={image_size}:{image_size}:0:0[a];"
        f"[1:v]fps=10,scale={image_size}:{image_size}:flags=lanczos[b];"
        "[a][b]psnr"
    )
    p = subprocess.run(
        ["ffmpeg", "-y", "-i", str(variant), "-i", str(raw), "-lavfi", graph, "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    m = PSNR_RE.search(p.stderr.decode())
    if not m:
        raise RuntimeError(f"psnr parse failed for {variant}: {p.stderr[-300:]!r}")
    psnr = float("inf") if m.group(1) == "inf" else float(m.group(1))

    graph_s = graph.replace("psnr", "ssim")
    s = subprocess.run(
        ["ffmpeg", "-y", "-i", str(variant), "-i", str(raw), "-lavfi", graph_s, "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    m = SSIM_RE.search(s.stderr.decode())
    if not m:
        raise RuntimeError(f"ssim parse failed for {variant}")
    return psnr, float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--variant-dirs", nargs="+", required=True)
    ap.add_argument("--sample-n", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="-")  # "-" = stdout
    args = ap.parse_args()

    raws = sorted(Path(args.raw_dir).glob("*.mp4"))
    if not raws:
        sys.exit(f"no mp4 under {args.raw_dir}")
    step = max(1, len(raws) // args.sample_n)
    sample = raws[::step][: args.sample_n]

    jobs = []
    for d in args.variant_dirs:
        d = Path(d)
        for raw in sample:
            v = d / raw.name
            if v.exists() and v.stat().st_size > 0:
                jobs.append((d.name, raw.name, v, raw))
    print(f"jobs={len(jobs)} (variants x completed sample files)", file=sys.stderr)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(measure, v, raw, args.image_size): (name, stem)
            for name, stem, v, raw in jobs
        }
        for fut, (name, stem) in futs.items():
            try:
                psnr, ssim = fut.result()
                rows.append({"variant": name, "stem": stem, "psnr_db": f"{psnr:.3f}",
                             "ssim": f"{ssim:.5f}"})
                print(f"{name} {stem} psnr={psnr:.2f} ssim={ssim:.4f}", file=sys.stderr)
            except Exception as e:
                print(f"FAIL {name} {stem}: {e}", file=sys.stderr)

    out = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    w = csv.DictWriter(out, fieldnames=["variant", "stem", "psnr_db", "ssim"])
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["variant"], r["stem"])))


if __name__ == "__main__":
    main()
