#!/usr/bin/env python3
"""Build the pack index for the YouTube crawl corpus (custom data layout).

Walks data/crawl/ref64/*.mkv (64px/10fps crf28 GOP20 references, frame-count
verified 1:1 against the 128px refs) and emits pack_index.npz:

    version        int32 scalar (1)
    fps            float32 scalar (10.0)
    image_size     int32 scalar (64)
    gop            int32 scalar (20)
    stem           U16 array     youtube id
    path           U256 array    absolute mkv path at build time
    n_frames       int32         exact decoded frame count (ffprobe)
    n_bytes        int64
    duration_s     float32       n_frames / fps
    minecraft      bool          crude title/channel filter (informational)
    channel        U64           from download meta ("" if unknown)

The train loader (src/pan2/data/gpu_pipeline.py, prefer_source="pack") reads
this for exact max_start per episode instead of probing/guessing lengths.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

VERSION = 1
FPS = 10.0
IMAGE_SIZE = 64
GOP = 20

# Crude Minecraft filter over the title (channel names deliberately NOT
# included: whole-channel harvests import non-Minecraft series, and a
# channel-listed regex would flag them all - the 2026-07-16 corpus scan
# measured ~39% non-Minecraft-looking titles this way). Informational flag;
# the sampler decides (pack_minecraft_only).
MC_RE = re.compile(
    r"minecraft|hermitcraft|foolcraft|vault hunters|skyblock|rlcraft|"
    r"create mod|modded|better minecraft|all the mods|atm[0-9]|enigmatica|"
    r"stoneblock|sevtech|pixelmon|hypixel|sky factory|dawn craft|"
    r"cobblemon|prominence|radium|crundee|dw20|direwolf|ftb|tekkit",
    re.I,
)

META_DELIM = "\\t"  # dl_worker writes literal backslash-t in list.tsv


def probe_frames(path: Path, ffprobe: str = "ffprobe") -> int:
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    s = r.stdout.strip()
    if r.returncode != 0 or not s.isdigit():
        return -1
    return int(s)


def load_meta(meta_tsv: Path) -> dict[str, tuple[str, str, int]]:
    """id -> (channel, title, duration_s) from the crawl meta TSV."""
    out: dict[str, tuple[str, str, int]] = {}
    if not meta_tsv.is_file():
        return out
    with open(meta_tsv, newline="") as f:
        for line in f:
            parts = line.rstrip("\n").split(META_DELIM)
            if len(parts) < 5:
                continue
            vid, channel, dur, title = parts[0], parts[1], parts[2], parts[3]
            if vid in out:
                continue
            try:
                out[vid] = (channel, title, int(dur))
            except ValueError:
                out[vid] = (channel, title, 0)
    return out


def build(
    ref64_dir: Path,
    out_path: Path,
    meta_tsv: Path,
    workers: int = 8,
    limit: int | None = None,
) -> dict:
    mkvs = sorted(ref64_dir.glob("*.mkv"))
    if limit is not None:
        mkvs = mkvs[:limit]
    meta = load_meta(meta_tsv)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        frame_counts = list(ex.map(probe_frames, mkvs))

    keep = [(p, n) for p, n in zip(mkvs, frame_counts) if n > 0]
    dropped = len(mkvs) - len(keep)

    stems = np.array([p.stem for p, _ in keep], dtype="U16")
    paths = np.array([str(p) for p, _ in keep], dtype="U256")
    n_frames = np.array([n for _, n in keep], dtype=np.int32)
    n_bytes = np.array([p.stat().st_size for p, _ in keep], dtype=np.int64)
    duration = (n_frames.astype(np.float32) / np.float32(FPS))
    channels = np.array(
        [meta.get(p.stem, ("", "", 0))[0] for p, _ in keep], dtype="U64"
    )
    titles = np.array(
        [meta.get(p.stem, ("", "", 0))[1] for p, _ in keep], dtype="U160"
    )
    minecraft = np.array(
        [bool(MC_RE.search(f"{c} {t}")) for c, t in zip(channels, titles)],
        dtype=np.bool_,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        version=np.int32(VERSION),
        fps=np.float32(FPS),
        image_size=np.int32(IMAGE_SIZE),
        gop=np.int32(GOP),
        stem=stems,
        path=paths,
        n_frames=n_frames,
        n_bytes=n_bytes,
        duration_s=duration,
        minecraft=minecraft,
        channel=channels,
        title=titles,
    )

    header = {
        "version": VERSION,
        "fps": FPS,
        "image_size": IMAGE_SIZE,
        "gop": GOP,
        "episodes": len(keep),
        "dropped_probe_failures": dropped,
        "total_frames": int(n_frames.sum()),
        "total_hours": float(n_frames.sum() / FPS / 3600),
        "total_bytes": int(n_bytes.sum()),
        "minecraft_fraction": float(minecraft.mean()) if len(keep) else 0.0,
        "meta_coverage": float(np.mean([c != "" for c in channels])) if len(keep) else 0.0,
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(header, f, indent=2, sort_keys=True)
    return header


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref64-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=None,
                    help="crawl meta list.tsv (literal backslash-t separated)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    header = build(
        args.ref64_dir,
        args.out,
        args.meta if args.meta else Path("/nonexistent"),
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(header, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
