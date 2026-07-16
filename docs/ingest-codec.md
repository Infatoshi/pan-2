# Codec ingest recipe (measured 2026-07-15)

Evidence base for turning scraped gameplay video into training shards without
storage becoming a project. All numbers measured on our 113.3 h corpus
(1625 episodes, VPT-style 640x360@20fps H.264 mp4). Artifacts:
`/data/pan-2/encode_ablation/`. Tooling: `scripts/measure_codec_quality.py`.

## Training view

Everything training consumes is defined by one reference transform:

```
raw mp4  ->  fps=10  ->  scale 128x128 lanczos   (= 128 px @ 10 FPS, Pan-1 res)
```

GOP 20 (2 s) for all encodes: fast window seeks, ~8% bitrate cost vs default
256-frame GOP. Passed straight into the ring loader via
`--prefer-source mp4 --raw-dir <variant dir>`.

## Measured rate-distortion

Full-corpus average bitrates; PSNR/SSIM on a 24-episode sample vs the
reference view above.

| variant | bitrate | PSNR | SSIM | TB per 500k h |
|---|---|---|---|---|
| x265 medium crf23 | 52 kbps | 43.3 dB | 0.978 | 11.8 |
| x265 medium crf28 | 29 kbps | 40.2 dB | 0.961 | 6.5 |
| x265 medium crf33 | 16 kbps | 37.4 dB | 0.936 | 3.5 |
| hevc_nvenc p7 cq20 | 81 kbps | 45.1 dB | 0.983 | 18.1 |
| hevc_nvenc p7 cq26 | 44 kbps | 41.9 dB | 0.969 | 9.9 |
| hevc_nvenc p7 cq32 | 23 kbps | 38.6 dB | 0.944 | 5.2 |

Anchors: raw uint8 npy at this view is ~3.9 Mbps (~885 TB per 500k h); the
VPT-source files as stored are 4.6 Mbps (~1.04 PB). Minecraft at 128 px
compresses to less than a podcast.

## Verdict

- **x265 medium beats NVENC p7/tune-hq by ~1.5 dB at equal bitrate**
  (~35-40% bitrate saved at equal quality). Part of NVENC's deficit is
  forced padding (see below).
- Storage is cheap either way: 500k h lands between 5 and 19 TB across this
  entire table. The encoder decision is CPU-fleet-time vs GPU-seconds, not
  a quality risk.
- **Quality-validated ingest recipe: x265 medium, crf28, fps=10, GOP 20**
  (29 kbps, 40.2 dB, 6.5 TB per 500k h).
- **Fast-ingest recipe: hevc_nvenc p7, tune hq, cq26, GOP 20** (44 kbps,
  41.9 dB, ~5-10x faster per stream). Use when crawl throughput dominates.
- crf33-class quality (37 dB, SSIM 0.936, worst episode 33.7 dB) is
  measurably soft; do not sign a dataset at that level to save $100 of disk.
- Before the real crawl fires, re-derive both settings on scraped
  third-person content (noisier than contractor footage, likely +30-100%
  bitrate; `measure_codec_quality.py` is the check).

## Gotchas (hard-won, permanent rules)

1. **Decimate with the `fps` filter, never the `-r` output option.**
   `-r 10` on these files produced dupped frames and half-frame drift
   (25.9 dB vs reference) in files that were bitrate-identical to healthy
   ones (38.9 dB). Size, bitrate, and content histograms all looked fine;
   only frame-aligned PSNR against a counted reference caught it. The first
   1625-episode x265 sweep was wiped and rebuilt. Frame-verify any new
   encoder path before bulk runs.
2. **NVENC rejects 128x128** (h264_nvenc and hevc_nvenc, 3090/driver 58x).
   Workaround: pad 128 content to 144x144 with black, crop back at decode
   (`crop=128:128:0:0`). Pad codes near-free but pollutes the RD comparison
   by a few percent against NVENC.
3. `-hwaccel cuda` decode was ~20% slower end-to-end than CPU decode at
   this resolution; the NVENC sweep runs CPU decode + ASIC encode.
   NVENC-only load means 4-8 concurrent sessions never disturb training.
4. Reference alignment check for a new encoder: fps=10+scale reference of
   the same source, frame count match, PSNR vs that reference > 35 dB at
   your target quality point.

## Open item: does any of this touch the training signal?

PSNR answers the video question, not the learning question. The staged
check (ready to run, shared seed via `build_state`):

```
arm A (control): shard npy path, configs/default.yaml
arm B: --prefer-source mp4 --raw-dir /data/pan-2/encode_ablation/crf28
arm C: --prefer-source mp4 --raw-dir /data/pan-2/encode_ablation/nvenc_cq32
```

Same Stage A config each; contrastive accuracy ramp at 400-1000 steps is
the readout. Our current task is scene-level discrimination, so "no
difference" shows codec loss doesn't hurt coarse signal; a sharper probe
(block-state / held-item pairs) is needed before trusting short-horizon
fine detail at crf33-class quality.
