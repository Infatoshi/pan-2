#!/usr/bin/env python3
"""Pack VPT-style data into pan2 shards (single ingest format).

Sources:
  episodes: read <stem>.img.npy/(.act.npy) directly (no re-encode, lossless).
  raw:      decode <stem>.mp4 via ffmpeg to image_size (e.g. 128px Pan-spec
            recode from 360p sources); act pulled from matching episode npy.

Every future acquisition (YouTube scrape, new contractor batch) should write
through ShardWriter too, so loaders never see per-source formats.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pan2.data.shards import ShardWriter


def decode_mp4(mp4: Path, image_size: int, ffmpeg: str = "ffmpeg") -> np.ndarray:
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(mp4),
        "-vf",
        f"scale={image_size}:{image_size}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    frame_bytes = image_size * image_size * 3
    if proc.returncode != 0 or len(proc.stdout) < frame_bytes:
        raise RuntimeError(
            f"ffmpeg failed {mp4.name} rc={proc.returncode} "
            f"bytes={len(proc.stdout)} err={proc.stderr[-200:]!r}"
        )
    n = len(proc.stdout) // frame_bytes
    return np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8).reshape(
        n, image_size, image_size, 3
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["episodes", "raw"], default="episodes")
    p.add_argument("--episodes-dir", default="/data/pan-2/episodes")
    p.add_argument("--raw-dir", default="/data/pan-2/raw")
    p.add_argument("--out", default="/data/pan-2/shards")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--target-shard-gb", type=float, default=4.0)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    eps_dir = Path(args.episodes_dir)
    raw_dir = Path(args.raw_dir)
    w = ShardWriter(
        args.out,
        image_size=args.image_size,
        target_shard_bytes=int(args.target_shard_gb * 1024**3),
    )

    if args.source == "episodes":
        imgs = sorted(eps_dir.glob("*.img.npy"))
        if args.max_episodes:
            imgs = imgs[: args.max_episodes]
        for i, img in enumerate(imgs):
            stem = img.name.replace(".img.npy", "")
            frames = np.load(img)
            act_path = img.with_name(img.name.replace(".img.npy", ".act.npy"))
            act = np.load(act_path) if act_path.exists() else None
            if args.image_size != 64:
                raise ValueError("episodes source is already 64px; use --source raw to recode")
            w.add_episode(frames, act, stem)
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(imgs)}] {stem}", flush=True)
    else:
        mp4s = sorted(raw_dir.glob("*.mp4"))
        if args.max_episodes:
            mp4s = mp4s[: args.max_episodes]
        for i, mp4 in enumerate(mp4s):
            stem = mp4.stem
            frames = decode_mp4(mp4, args.image_size)
            act_path = eps_dir / f"{stem}.act.npy"
            act = np.load(act_path) if act_path.exists() else None
            if act is not None and act.shape[0] != frames.shape[0]:
                # trim to shorter stream; jsonl/act and mp4 can disagree by a frame
                t = min(act.shape[0], frames.shape[0])
                frames, act = frames[:t], act[:t]
            w.add_episode(frames, act, stem)
            if (i + 1) % 25 == 0:
                print(f"[{i + 1}/{len(mp4s)}] {stem}", flush=True)

    header = w.close()
    gb = header["total_frames"] * (args.image_size**2) * 3 / 1024**3
    print(
        f"done: episodes={header['total_episodes']} shards={header['n_shards']} "
        f"frames={header['total_frames']} ({header['total_frames']/20/3600:.1f}h @20Hz) "
        f"frames_bytes={gb:.1f}GiB -> {args.out}"
    )


if __name__ == "__main__":
    main()
