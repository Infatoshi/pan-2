# Custom data layout: pack (crawl corpus)

Design goals (user, 2026-07-16): small on disk, decompressed on the fly,
training throughput as high as the GPU allows. This doc is the layout
contract; `docs/ingest-codec.md` covers codec RD evidence.

## Layout

```
data/crawl/
  raw/<id>.<ext>            360p avc1 originals (29.97/30fps), source of truth
  ref/<id>.mkv              128px/10fps x265 crf28 GOP20 reference view
                            (frame-count verified vs duration*10)
  ref64/<id>.mkv            64px/10fps x265 crf28 GOP20, derived 1:1 from ref/
                            (EXACT frame-count equality gate) - train reads this
  pack/pack_index.npz       episode index v1 (scripts/build_pack_index.py)
  pack/pack_index.json      human header (counts, hours, fractions)
```

Index v1 fields: stem, path, n_frames (exact, ffprobe-counted), n_bytes,
duration_s, minecraft (crude title-regex flag, informational), channel,
title; scalars version=1, fps=10.0, image_size=64, gop=20.

## Loader contract (prefer_source="pack")

- Seeks are exact-frame: `-ss` before `-i` with decode+discard, verified
  bit-exact against sequential decode at GOP boundary straddles
  (tests/test_pack_loader.py).
- Decode at native size, no in-pipe scale: index image_size must equal
  model image_size (64); the loader raises otherwise.
- Index fps must equal cfg.native_fps (10.0); goal horizons are native
  frames (10..150 = 1s..15s, same wall contract as 20..300 @20Hz).
- Stride applied exactly once: the 29.97/30 -> 10fps fps-filter at ingest
  is the only subsample; pack configs run frame_subsample=1 (ring k=1,
  model k=1).
- max_start is exact from the index (no probing, no guessing).
- The loader loads the item list once at startup; rebuild the index
  (scripts/build_pack_index.py) and restart to pick up new episodes.
- ref64 tail frames (< t_load per episode) are unreachable; negligible.

## Why this shape (benches 2026-07-16, all on real corpus refs)

Per-window random access caps every engine: PyAV seek+to_ndarray 1.1k fps
(rgb conversion costs ~7x the decode itself), torchcodec CPU 6.6k,
torchcodec NVDEC 4.0k. NVDEC is unusable for v1: rejects streams narrower
than 144px ("width 128 not within range from 144 to 8192"), and even at
144px every ffmpeg/torchcodec path syncs per frame at 7-12k fps with no
multi-stream scaling. Sequential ffmpeg decode is fast instead: 11-26k
fps/instance at 128px (frame-threaded), ~2.5x cheaper per frame at 64px.

Ring math then decides: at refresh_prob 0.05 the GPU clip ring only needs
~146 fresh slots/s x t_load(428) frames = ~54k decoded fps - inside the
64px CPU envelope with ~12 stateless window-decode producers. Entering
persistent pipes, bare annexb streams, or GPU decode buys nothing at this
demand, so v1 stays stateless per-window ffmpeg rgb24, reusing the proven
project code path.

## Measured E2E (2026-07-16, GPU0, lease)

configs/pretrain_pack_preview.yaml, bs=64, context 128 @10fps k=1, 1000
steps, 81 episodes (34.1h): 51-54 ms/step over steady state (8,320
slot-frames/step) = 4.44 h-video/s vs 4.80 h/s shard-path at bs=64
conv-on - 8% delta attributable to co-running transcode CPU load (12
producers + 10 x265 during the run). errors=0, ~73 fills/s sustained,
min-fill 338/6721 in ~15s, loss 1.10 -> 0.010 @1000 steps (same family
of curve as the codec arms).

## Stored-size math

ref64 measured 8.6MB/h (3.2MB per median episode) -> ~39GB per 4,400h
corpus; index ~2KB per 1k episodes. raw 1.10TB and ref-128 ~87GB are
kept as source-of-truth and the 128px hedge respectively.

## v2 candidates (with the numbers that would justify them)

- Async NVDEC at 144px: 4 engines, paper rate ~500Mpx/s each; requires a
  real async surface-queue client (ffmpeg/torchcodec sync APIs measured
  7-12k fps = dead). Re-encode cost: full corpus at 144px (~6h at 12-wide).
- nvJPEG batch decode: needs JPEG storage (~5x ref64 size, ~200GB per
  4,400h) plus binding work; only if CPU decode becomes the limiter.
- Either swap is isolated behind the producer interface; consumers don't
  change.
