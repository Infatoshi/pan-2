# DEVLOG: pan-2

## 2026-07-15 — project scaffold

- Created anvil-primary repo at `~/dev/pan-2`.
- Method: Pan-style goal-conditioned pretrain + action post-train; single Blackwell.
- Existing data on box: `/data/vpt/raw` (~241G), `/data/vpt/episodes` (~95G).
- Disk free (approx): `/` ~2.1T, `/data` ~1.4T. User handling additional space/data on the side.
- Kernel SOTA assumed non-blocking; architecture favors cheap frame tokens + chunked actions + swap-friendly temporal backbone.
- Next: keep synthetic smoke green; wire real VPT loader sanity after deps; scale data when user frees space.

## 2026-07-15 — smoke green

- `uv sync --extra dev` OK (torch 2.13 + cu13).
- `uv run pytest -q`: 4 passed.
- `uv run ruff check . --fix`: clean.
- `CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke.py`: pretrain+posttrain train steps OK.
- VPT episode probe: `img.npy` is `(T, 64, 64, 3) uint8`, `act.npy` is `(T, 25) float32`. Loader already resizes to `image_size`; action layout still heuristic (last-2 mouse) and needs a proper VPT action map before real post-train.
- GPU0 had ~55GB in use at scaffold time; still enough for smoke. Check `nvidia-smi` before larger runs.


## 2026-07-15 — VPT data cleaned and moved

- Source `/data/vpt` validated and moved to `/data/pan-2`.
- Kept 1625 good episodes (8155382 frames, ~113.3h @ 20Hz). Dropped 27 (too_short / empty img).
- Layout: episodes/, raw/ (good only), meta/, quarantine/.
- Project symlink: `~/dev/pan-2/data/vpt` -> `/data/pan-2`.
- Defaults: `data_dir=/data/pan-2/episodes`, `n_discrete=23` (VPT act dim 25 = 23+2 mouse).

## 2026-07-15 — pretrain infra perf (real /data/pan-2/episodes)

Harness: `scripts/bench_pretrain.py`. Default-ish: d_model=512, 8 layers, ~28.5M params, bf16, pretrain contrastive.

### RTX 3090 (GPU1), bs=16, T=128, img=128
- getitem ~2.1ms (slice cheap; float+resize ~1ms)
- dataloader alone: nw=0 best ~9.2 batch/s; nw>0 slower (mmap/worker overhead)
- gpu_compute_only: 276 ms/step
- h2d: 68 ms, **403 MB/batch** float32 frames
- full wall: **331 ms/step** (~3.0 step/s, ~6.2k frames/s)
- phase share: backward 47.6%, forward 33.0%, h2d 17.2%, data_wait 0.1%
- **BOTTLENECK: GPU compute (backward)**; secondary H2D from float32 pixels

### 3090, bs=16, T=128, img=64 (native)
- full wall: **108 ms/step** (~9.3 step/s); still GPU-bound (~50% bwd)

### Blackwell GPU0 (contended ~55GB other job), bs=8, T=64, img=64
- full wall: **16.6 ms/step** (~60 step/s); still GPU-bound (bwd 44%)

Fixes ranked: (1) uint8 H2D + GPU normalize (2) train native 64 or fuse resize on GPU (3) larger microbatch on free Blackwell (4) kernel/backbone later.


## 2026-07-15 — pretrain infra opts + rebench

Implemented:
1. uint8 H2D + GPU normalize (`prepare_images`)
2. native 64px default (no CPU upsample)
3. larger batch defaults (32/64) + optional `torch.compile`

Blackwell GPU0 (contended ~55GB other job), real `/data/pan-2/episodes`:

| config | wall ms/step | frames/s | serial cpu/xfer/gpu |
|--------|-------------:|---------:|---------------------|
| old-ish float host, bs16, want 128 | 87 | 23k | 1.3 / 4.4 / 94.3 |
| uint8 native64 bs32 | 49 | 83k | 1.8 / 4.1 / 94.1 |
| uint8 native64 bs64 | 92 | 89k | 0.3 / 4.8 / 95.0 |
| uint8 bs64 + compile | 69 | 119k | 2.3 / 5.6 / 92.1 |

Bottleneck remains GPU kernels (~92-95% serial share).


## 2026-07-15 — GPU kernel focus (SDPA temporal)

Replaced `nn.TransformerEncoder` with custom pre-norm blocks + `scaled_dot_product_attention(is_causal=True)`.
Also: TF32, cudnn.benchmark, channels_last CNN, fused AdamW.

### 3090 results (uint8, 64px, T=128, free GPU1)

| stack | wall ms | frames/s | gpu% serial |
|-------|--------:|---------:|------------:|
| prior (nn.TransformerEncoder) bs32 | 185 | 22k | 95 |
| **SDPA bs32** | **86** | **47.5k** | 91 |
| SDPA+compile bs32 | 129 | 32k | 92 (compile hurt wall) |
| SDPA+compile bs64 | 157 | 52k | 89 |

Winner for steady train: SDPA without compile at bs32 (~2.15x vs previous optimized path).
compile: false default; optional for large-batch experiments.

## 2026-07-15 — CNN hotspots fixed + reprofile (3090)

Implemented from profiler plan:
1. BatchNorm -> GroupNorm
2. Cheaper encoder (stem 32 + depthwise separable blocks)
3. Temporal frame_subsample=4 (dataset + model; keep last frame)
4. Subsample before cast; uint8 H2D only for T/k frames

### 3090 real-data profile (bs32 T128 img64)

| | wall ms | top-1 |
|--|--------:|-------|
| before (SDPA + full CNN BN every frame) | 85.8 | conv_bwd 25% |
| after | **20.0** | copy_ 17% |

Top5 after: copy_ 17%, HtoD 10%, group_norm 8%, conv_bwd 7%, mm 6%.
~4.3x step speedup.

## 2026-07-15 — GPU pipelined loader (~10GB ring)

Added `pan2.data.gpu_pipeline.PipelinedGpuPretrainLoader`:
- GPU uint8 ring sized by budget_gb (default 10)
- async producers: npy (fast) or mp4/ffmpeg (on-the-fly) -> pin -> H2D
- train path only index_select on GPU (data_only ~0.1ms warm)

3090 bench (bs32, T_sub=33, stem32, GN encoder):
- auto/npy fill: 8.6GB ring, fill 0.6s to ready, data_only 0.09ms, wall ~31ms/step
- mp4 fill: works; slower warm (~5s for 2GB); short clips can underrun ffmpeg frames
- NVDEC scale_cuda not usable on this ffmpeg build; CPU scale@64 is faster for short windows

Train entry: `scripts/train_pretrain_pipeline.py --budget-gb 10`

## 2026-07-15 — correctness pass: goals, action layout, subsample bugs

Triggered by an external review that (correctly) flagged the pipeline as
optimizing a broken task. Fixes, all measured not guessed:

1. **Goal sampling was copy detection.** Old: goal = last context frame
   (dataset) or random in-clip frame (gpu_pipeline). Fixed: goal = frame
   strictly after context end, horizon ~ U(20, 300) native frames (1..15s at
   20Hz) in both loaders (`min_goal_horizon` / `max_goal_horizon` in
   TrainConfig/PipelineConfig). gpu_pipeline slots now store t_sub context
   frames + 1 baked future-goal frame; used slots recycle with p=0.05 so the
   ring refills fresh clips/goals (before, recycling was never called and the
   ring served a frozen capacity-sized subset; added producer->default stream
   ordering to make recycling race-free).

2. **Action layout recovered from data.** act dim 25: cols 0-22 binary,
   23-24 camera dx/dy quantized to 0.1 steps in [-1, 1] (21 bins). Column to
   key mapping recovered by correlating act columns with raw jsonl key events
   (frame-aligned recall/precision over 10 episodes): 0=esc 1=s 3=w 13=e
   14=space 15=a 16=d 17=lshift 18=lctrl 20=mouse.0 21=mouse.1; 12 columns
   dead (presumed hotbar). Authoritative table in `pan2/actions.py`.

3. **Double subsampling bug.** VPTEpisodeDataset subsampled windows at k AND
   PanPolicy subsampled again at k (pipeline path avoided it by forcing model
   k=1). At k=8 the plain train path encoded 3 tokens per clip. Fixed: dataset
   returns full-rate windows; the model is the single subsampler (pipeline
   subsamples at ring fill, model k=1 as before).

4. **frame_subsample default 8 -> 2.** Data is 20Hz; k=2 gives 10fps tokens
   (Pan rate). k=8 (2.5fps) makes the next-10-action chunk (0.5s) invisible
   to the model and starves context. All yaml configs pin k=2. T=128, k=2 ->
   65 tokens/clip.

5. Dataset `__len__` no longer multiplies epochs by 8; synthetic fixtures now
   use a future-frame goal (plumbing shape matches the real task; synthetic
   stays unlearnable by construction, it validates shapes not accuracy).

Fixed-task learnability (3090, 27M model, pipelined loader, bs=32, 400 steps,
real data): contrastive top-1 retrieval 0.40 -> 0.97 vs chance 0.031. Note:
in-batch negatives are mostly other clips, so this is largely scene matching;
harder negatives (same-episode, other horizons) are the next task upgrade
before claiming goal-directed representations.

Perf after fixes (3090 GPU1; GPU0 busy with another job, no contended numbers
reported). Config: 27M params, T=128, k=2 (65 tok/clip), img64, bf16, bs=32:
- mmap dataset path (train_pretrain.py): wall 40.1 ms/step, ~102k frames/s,
  serial share cpu 6.9 / h2d 18.3 / gpu 74.8 (h2d up because windows are now
  full-rate; pipeline path avoids this)
- pipelined ring path (train_pretrain_pipeline.py): wall 31.0 ms/step,
  1032 clips/s, ~67k tokens/s, per-token-throughput within noise of the
  pre-fix k=8 kernel rates (0.0149 vs 0.0145 ms/clip-token), i.e. the kernel
  work had converged; what changed is the task being fed is now correct.
- ring refill verified: fills > capacity, 0 errors, refresh cycling.

Tests: 11 passed incl. new future-goal regression tests; ruff clean (also
fixed leftover lint debt in profile/sweep scripts).

## 2026-07-15 — re-profile after CNN prune (subsample 8, stem/b1 GN gone)

Opts already landed: `frame_subsample=8` default; stem GN removed; block1 is single stride-2 conv (no GN / no DW+PW); GPU ring pipeline.

**Unit:** `uv run pytest -q` → 9 passed (1.35s).

**GPU:** 3090 free (CUDA_VISIBLE_DEVICES=1). GPU0 Blackwell busy ~57GB / 100%.

### Module conv/GN (`profile_conv_gn_bwd.py`, B=256, budget 3GB ring, model subsample=1)

- wall avg **67.4 ms** (p50 63.5)
- tracked modules: stem.conv, b1_conv, b2_dw/pw/gn, b3_dw/pw/gn (no stem.gn, no b1 gn/dw)

| aggregate | ms | % wall |
|-----------|---:|-------:|
| all_conv_fwd | 7.3 | 10.8% |
| all_gn_fwd | 2.4 | 3.5% |
| all_conv_bwd | 11.3 | 16.8% |
| all_gn_bwd | 3.9 | 5.8% |
| **conv_bwd+gn_bwd** | **15.3** | **22.6%** |

Stage bwd rollup:
- stem: 0.0% wall (hook; stem input no grad — stem still shows in aten conv_bwd shapes)
- block1: **4.9 ms / 7.3%** (b1_conv only)
- block2: 4.7 ms / 7.0% (conv 3.3 + gn 1.4)
- block3: 5.6 ms / 8.3% (conv 3.1 + gn 2.5)

Per-module bwd leaders: b1_conv 7.3%, b2_dw 3.9%, b3_gn 3.7%, b3_dw 3.3%, b2_gn 2.1%.

### Pipeline full step (`bench_gpu_pipeline.py`, B=256, T=128→17 uint8, subsample 8 on ring)

- full_step_wall **68.8 ms** (~14.5 step/s)
- gpu_compute_only 62.5 ms; data_only 0.74 ms; stall 6.4 ms (9% wall)
- frames shape `(256, 17, 3, 64, 64)` uint8

### Train-step top kernels (`profile_train_step.py` on-device, B=256, model subsample=8)

- wall avg **64.2 ms** (p50 64.2)

Top 5 (self device, % of wall):
1. `aten::convolution_backward` — 18.0 ms, **28.0%**
2. `aten::mm` — 9.5 ms, **14.8%**
3. cutlass_gemm_or_fprop — 7.5 ms, **11.7%**
4. group_norm_kernel — 7.3 ms, **11.3%** (remaining b2+b3 GN only)
5. gelu_backward_kernel — 6.5 ms, **10.1%**

### vs previous baseline (this session, pre-prune round)

| metric | before | now |
|--------|-------:|----:|
| wall @ B=256 | ~61 ms | **64–69 ms** (on-device 64 / module 67 / pipeline 69) |
| conv_bwd+gn_bwd % wall | ~38% | **22.6%** (module hooks) |
| stem.gn | ~8% | **gone** |
| b1_dw | ~7.5% | **gone** (replaced by b1_conv ~7.3% bwd) |

**Read:** GN surface area cut worked (stem+b1 GN out; aggregate conv+GN bwd share 38%→23%). Wall did **not** improve and may be slightly worse — block1 full 3×3 s2 conv (~7% bwd) is a similar-cost replacement for the old DW path, and remaining hot spots are stem/b1 dense conv_bwd, transformer mm/GEMM, and residual b2/b3 GN. Next leverage is likely attention/MLP (mm + gelu_bwd ~25% combined) or cheaper stem/b1 spatial path, not more GN deletion alone.

## 2026-07-15 — same-episode hard negatives (task hardening)

Loaders now also return `neg`: a frame from the SAME episode strictly past
the goal window (neg_idx = context_end + max_horizon + 1..max_horizon).
Extra column in `GoalValueHead.logits` (own-row hard negative) aimed at
killing the scene-ID shortcut left by cross-episode in-batch negatives.
Ring slots are now [ctx toks | goal | neg]; `train.hard_negatives` toggles.

400-step rerun (3090, pipelined, bs=32, real data, fixed task + hard neg):
- acc(last-50): 0.33 -> 0.95 by step 400 (chance 0.030); without hard neg the
  same run hit 0.97. Harder, learnable, still not saturated-hard; candidate
  next step: negatives from INSIDE the goal window at wrong horizons.
- wall: 36.6 ms/step vs 31.0 without hard neg (extra frame encode per row).

## 2026-07-15 — open-source restructure

- Public packaging: pyproject 0.2.0 (MIT, classifiers, urls), LICENSE added.
- `pan2/kernels/` created as the custom-op home: registry + contract
  (ref impl + optimized impl + unit test + bench per op, claims cite benches).
- README rewritten as public intro (method, data contract, quickstart, layout,
  pitfalls). CLAUDE.md/AGENTS.md harmonized on the new layout. SPEC gains the
  kernel contract + hard-negative objective spec.
- Gates at restructure: pytest 12 passed, ruff clean, smoke green.

## 2026-07-15 — posttrain overfit sanity (SPEC criterion 2) + epoch bug

`configs/posttrain_overfit.yaml`: 1 real episode, d256/4L, T=64, k=2, bs=16,
500 steps (3090). discrete_bce 0.185 -> 0.046, mouse_mse ~0.037 -> ~0.03-0.04
(noisy), total loss 0.31 -> 0.09. Action path fits real measured-layout
labels; note ~12/23 button columns are dead so early BCE is easy to deflate.

Bug found: removing the `len(pairs) * 8` epoch multiplier broke tiny
datasets: max_episodes=1 gave len(ds)=1 < bs=16 with drop_last=True, so the
DataLoader produced zero batches per epoch and infinite_loader churned
worker fork/join forever (diagnosed via py-spy: stuck in _shutdown_workers).
Fix: explicit `windows_per_episode` knob (default 64), documented as the
epoch definition. Also: always run training scripts with `python -u` (or
accept pipe-buffered logs).

## 2026-07-15 — intent recovered: this is a Pan-1 single-GPU repro

The Grok session's first prompt surfaced: repro pantograph.com/journal/pan-1
on one GPU. Read the journal (verified 2026-07-15). Key facts vs our repo:
- Pan-1: 128x128 @ 10FPS, 300-frame context, 20Hz actions, 9 keys + 2 mouse.
  Pretrain: hindsight in-context value fn + NEXT-FRAME DISTRIBUTION head.
  Data: 500k h video + 2k h contractor. Eval: 104-env grader suite, 30s cap.
- Our v0 goal mechanism (strictly future) matches their V(s,g) definition;
  what we lack is (a) the second (next-frame) objective, (b) cross-policy
  goal/context pairs (their corpus is arbitrary gameplay, ours is
  self-centred), (c) ~4 orders of magnitude of video data and most of the
  demo data, (d) any env.
- Verdict: kernel/microbench rounds (incl. this week's) optimized the least
  binding constraint. SPEC gains "Stage A v1: Pan-1 alignment" + a data
  acquisition section with ordering. README now states the repro mission and
  the honest gap table. Model work ahead of the data gap is premature.

## 2026-07-15 — shard format: single ingest for both stages

Built the packed shard pipeline and populated /data/pan-2/shards.

- Format (`pan2/data/shards.py`, v1): manifest.jsonl header + segment rows;
  shard-NNNNN.frames.npy uint8 memmap + .act.npy float32, shared frame
  offsets. Episodes never straddle shards. `ShardWriter` rejects mixed
  act/no-act builds eagerly and validates shape/size/dtype per episode.
- `pan2/data/windowing.py`: the sampling contract (window span, future goal
  gap, hard-neg gap, action chunk start) extracted so VPTEpisodeDataset and
  ShardDataset cannot drift apart. Selector `pan2/data/build.py` returns
  ShardDataset iff data_dir has a manifest.
- Ring loader: prefer_source gains "shard" (auto prefers shards when a
  manifest exists); producers mmap-slice at segment offset. Train scripts
  (pretrain/posttrain) and bench_pretrain route through the selector;
  bench auto-detects format from --data.
- Full build: 1625 episodes -> 24 shards, 8,155,382 frames = 113.3h,
  93.3GiB frames bytes, 95G on disk. Byte-identity spot check vs the source
  episode pair passes; ring-over-shards: 1546 usable items, 0 fill errors,
  0.50 ms/batch steady reads. Posttrain overfit through the shard path
  (GPU1): loss 0.36 -> 0.12 over 200 steps, matching the npy-path numbers.

Note: this is repackaging, NOT acquisition. The 500k h video gap stands.
The build script's --source raw branch recodes from the 640x360 mp4s and is
the funnel for new scraped video (also enables a true 128px build).

## 2026-07-15 — codec ablation: 500k h fits on one drive; x265 beats NVENC per bit

Re-encoded the full 113 h corpus to training view (128x128 @ 10fps,
GOP=20) across 6 variants and measured PSNR/SSIM vs the clean
downscale+decimate reference (24-episode sample,
`scripts/measure_codec_quality.py`).

- x265 medium: crf23 52 kbps 43.3 dB | crf28 29 kbps 40.2 dB |
  crf33 16 kbps 37.4 dB. Corpus totals: 6.5 TB per 500k h at crf28.
- hevc_nvenc (p7/tune-hq, 128 content padded to 144x144 because neither
  h264 nor hevc NVENC accepts 128x128 on the 3090): cq20 81 kbps 45.1 dB |
  cq26 44 kbps 41.9 dB | cq32 23 kbps 38.6 dB. Roughly 5-10x faster per
  stream than x265.
- Verdict: x265 medium is ~1.5 dB better than NVENC p7 across the curve
  (~35-40% bitrate saved at equal quality; NVENC's pad pixels are a small
  systematic against it). NVENC cq26 (42 dB, ~10 TB per 500k h) is the
  fast-ingest recipe when crawl throughput dominates.

Bug caught by the quality check (bitrate-invisible): `-r 10` as an output
option corrupted decimation on these files (dupped frames, half-frame
drift, 25.9 dB vs reference at identical size). `fps=10` as a filter is
frame-accurate (38.9 dB control). Rule now: decimate with the fps filter,
never -r, and frame-verify any new encoder path against a counted
reference before bulk runs. The first x265 sweep (which used -r) was
wiped and rebuilt.

Also: build_state now actually applies train.seed (was a dead knob);
train_pretrain_pipeline.py takes --raw-dir/--episodes-dir so A/B/C runs
over codec variants share seed and differ only in data.

## 2026-07-15 — measured bottleneck map (ring path is 100% GPU-bound)

Config: 27M params, T=128 (k=2, 65 tok/clip), bs=32, bf16, shard data,
steady state. Method: hard-sync phase timing + torch.profiler CUDA time.

End-to-end (GPU1 3090):
- DataLoader path: wall 38.2 ms/step = GPU 30.8 (79%) + H2D 7.8 (20%) +
  CPU 0.2 (0.5%, fully hidden by prefetch).
- Ring path (production): wall 30.9 ms/step, data-wait 0.1 ms (~0%).
  fwd 10.0 / bwd 20.7 / optimizer 0.5 ms.

GPU kernel buckets, 3090 (30.7 ms/step profiler total):
- transformer linear/MLP gemms 25.5%
- elementwise + activations + copies 21.7%
- conv backward (incl. depthwise) 19.4%
- NCHW<->NHWC layout conversion 8.0%
- GroupNorm fwd+bwd 6.1%
- conv forward 5.5%
- AdamW fused 2.9%, other 10.7%

GPU0 (RTX PRO 6000 Blackwell) ring wall: 10.9 ms/step (91.5 steps/s) =
2.8x the 3090 with identical code/config; memory headroom fits budget_gb
8 ring (10433 clips). Phased sums exceed wall due to sync-drain; wall is
the number to quote.

Kernel opportunities, in order: (a) kill the 8% layout churn
(channels_last end-to-end or Conv weights pre-transposed), (b) fuse
GN+bias+act in the encoder (another ~6%), (c) elementwise/act fusion in
the transformer blocks. Encoder conv stack fwd+bwd is ~25% but healthy
gemms after layout; transformer middle is memory-bound at T=65.

## 2026-07-15 — dataloader headroom: measured ceilings (it is not the bottleneck)

Question posed: "dataloader should NOT be the bottleneck." Measured rather
than assumed (GPU0, shard source, steady state):

- Per-step data cost on the ring path: 0.10 ms/fetch against a 30.9 ms
  (3090) / 10.9 ms (Blackwell) step = ~0%. Both GPUs sit at 100%
  GPU-bound walls.
- Producer fill rate: 689 clips/s aggregate at 8 producers, 644 at 16 —
  the limit is per-clip CPU cost (mmap read + pin + H2D), not threads.
- At Blackwell consumption pace (91.5 steps/s * bs 32 = 2930 clips/s
  equivalent), refresh_prob must stay <= ~0.235 sustained; the default
  0.05 carries ~5x headroom. The DataLoader (non-ring) path keeps its
  20% H2D share; that path is for benches/unit runs, not throughput.
- Added loader.status()["ready_low"]: low-watermark of resident clips
  between samples. If it ever trends toward 0, data is becoming the
  bottleneck; the ceiling math to respond is in gpu_pipeline.py's
  refresh_prob notes.

## 2026-07-15 — kernel batch 1: encoder fusion + transformer fusion (merged)

Two delegated worktrees (codex kA encoder, grok kB transformer), each
reviewed + acceptance re-run independently before merge. All numbers
production config: ring loader (k=2 at fill, model k=1, 65 ctx tokens +
goal/neg), bs=32, bf16 autocast, shard source.

kA (96156aa, GPU0 RTX PRO 6000): channels_last encoder end-to-end +
Triton fused GroupNorm+GELU (fp32 stats, exact erf). Fused op fwd
8.9x / fwd+bwd 9.2-9.3x at both production shapes [2080,128,8,8],
[2080,512,4,4]. Encoder fwd+bwd 6.40 -> 3.33 ms (-48%); NCHW<->NHWC
layout kernels 117 calls / 8.96 ms -> 0 (profiler-verified).

kB (335db8b, GPU1 RTX 3090): Triton bias+GELU (fwd 2.06x, fwd+bwd
1.07x at [32,67,2048] bf16), qkv unbind/reshape copy elision, and
torch.compile(mode=reduce-overhead) scoped to TransformerTemporal
(PAN2_TEMPORAL_COMPILE=0 kills it). Temporal wall 11.04 -> 9.02 ms
(-18.3%); ELEM_COPY profiler bucket 3.30 -> 1.86 ms (-43.6%). Numerics
vs eager: fwd max|diff| 4e-6, grads 9.5e-5 (fp32 probes).

End-to-end wall (60-step mean, same-session A/B):
- GPU0: 10.93 (baseline) -> 8.72 (kA) -> 7.45 (kA+kB) ms/step,
  -31.8% total; 91.5 -> 132 steps/s.
- GPU1: 33.80 (baseline) -> 29.07 (kB) ms/step (-14.0%).

Measurement caveats caught this round: (1) an intermediate "baseline"
18.2/17.0 ms pair on GPU1 was a harness bug (model k=2 stacked on ring
k=2 halved tokens to 33) discarded and re-measured at production wiring;
(2) the earlier 30.9 ms 3090 figure did not reproduce same-session
(33.8 baseline today) — drift ~3 ms across sessions, so only
same-session A/B deltas are quoted as earned.

Follow-up fix on main: save/load now strip/re-insert compile's
._orig_mod. key segments so checkpoints round-trip between compiled
and eager builds either direction (tests/test_ckpt_compile_compat.py).

## 2026-07-15 — post-merge bottleneck map (kA+kB stacked)

Re-profiled the production train step (ring k=2/model k=1, bs=32, bf16)
with scripts/profile_step.py; hand-classified kernel names (flash
attention kernels carry "cutlass" in template args, so regex buckets
miscount attention as gemm if classified naively). Wall GPU0 7.25
ms/step (kernel self-time 6.56; profiler-run drift band vs the 7.45 e2e
run). GPU1 stacked wall 21.2 ms/step (kernel self 21.1) = -37% vs the
33.8 same-session baseline.

Bucket shares of kernel self-time, GPU0 / GPU1:
- gemm 22.0% / 29.7% — now the top GPU0 bucket.
- conv fwd 18.6% / 12.6%, conv bwd 17.2% / 21.0% — cudnn NHWC; single
  largest kernel is one convolve_common_engine at 0.63 ms (9.6%).
- eager GELU fwd+bwd 10.0% / 8.0% — policy/heads sites kB did not
  cover; next fusion target.
- aten pointwise/copies 7.3% / 6.0% — bf16 casts, direct_copy, adds.
- adamw+foreach 7.2% / 4.4%.
- attention (flash) 3.3% / 4.8% — small at T=67.
- gn+gelu (kA Triton kernel) 3.9% / 3.8%; eager norms 4.0% (heads LN).
- inductor fused pointwise+LN ~4.0% combined; memcpy/memset ~2.3%.
- layout NCHW<->NHWC: 0 calls (was 8.0% / 117 calls per step).

Attack order from here: (a) heads GELU+copies fusion (~9% worst-case),
(b) conv bwd (dgrad+dw) — cudnn is already on NHWC engines, so the
lever is algo/autotune, not layout, (c) gemm efficiency — cublas is
serving sm80 cutlass binaries on sm_120; a torch build with sm120
cublas kernels may move the 22% bucket without code changes.

## 2026-07-15 — NCU on sm_120: where the step time actually goes

Isolated per-op harness (scripts/ncu_hot_kernels.py, production shapes
N=2080 encoder / 2144 temporal rows, bf16, channels_last, cudnn
benchmark on) + NCU SpeedOfLight/ComputeWorkloadAnalysis/Occupancy per
kernel class on GPU0 (sm_120), GPU1 (sm_86) as native-binary control.
12 parallel-safe processes (one per op per GPU; NCU serializes per GPU
internally). Empirical BF16 roof from giant cublas gemms: ~400 TF/s
(GPU0), ~71-75 TF/s (3090, matches spec).

Findings, in attack order:

1. cudnn channel-padding churn, ~1.3+ ms/step (~19% of GPU0 step).
   In-step nhwcAddPaddingKernel runs 6x/step at ~0.21 ms each. Root
   cause found via isolated conv_stem: the stem conv (Cin=3) is padded
   to 128 channels (template int 128) so cudnn can run an aligned NHWC
   engine - ~43x data amplification. Captured window: 7x AddPadding
   (1744us total) vs 1x actual convolve (1.4us, SOLc 78%). The conv
   itself is fine; the padding around it is the step. 3090 shows the
   same disease (tensorTransformGeneric 336us + AddPadding 245us).
   Fix: hand NHWC direct conv for Cin=3/32 (stem+b1 fusion), no pad.
2. Transformer gemms wave-quantized at ~50% SOL (bucket = 22% of
   step). cutlass_80 sm80 binaries on sm_120: SOL compute 52/52/42/31%
   on fc1/fc2/qkv/proj, achieved occupancy 8-11%, grids ~0.7 waves on
   188 SMs (M=2144 with 128-tiles). Not a per-SM problem as much as a
   too-few-CTAs problem. Fix: split-k or smaller/persistent tiles (hand
   gemm), or simply more tokens per device (batch up) - free SOL.
3. Eager erf-GELU is near memory SOL already (SOLm 89%, occ 88) - the
   kernel is fine, the WIN is not materializing: 0.66 ms/step of
   read/write passes that disappear if fused into neighbours (conv
   epilogue / GN kernel for stem+b1 which never got the kA fusion).
4. gn_gelu (ours): SOLm 36% at us-scale; healthy enough, revisit later.
5. AdamW multi_tensor 0.47 ms/step (7.2%): generic foreach kernel; a
   fused single-pass AdamW for 27M params should halve it. Cheap win,
   low risk, do after 1-3.

Projected headroom on GPU0: 1+2+3 recover ~2.5-3 ms of the 7.25 ms
step; floor is then set by convs-proper + gemm-proper at ~4.2-4.7
ms/step unless batch size rises (which raises gemm waves for free).

## 2026-07-15 - kernel batch 2: fused AdamW + fused gather/scale/cast

Two fronts from the post-merge bucket map landed this round (kC conv
frontend still in flight at time of writing).

AdamW (kD, delegated, merged a171222). Triton multi-tensor AdamW over a
pointer table (params keep storage/identity; state_dict schema matches
torch AdamW fused for round-trip both directions; PAN2_FUSED_ADAMW=0
opts out). Acceptance re-run by us, not the delegate: 54 passed in the
kD tree, 67 passed merged; bench_adamw.py on the 3090 reproduces exactly
(torch fused 1.061 ms, ours 0.999 ms mean over 200 iters; 1.06x; vs
foreach 2.79x). Roofline honesty note: 27M fp32 params x 7 streams is
~0.81 ms at the 3090's 936 GB/s, so torch fused was already at the
floor; the 1.5x target was unphysical and the delegate said so with
numbers instead of chasing it. E2E wall on the 3090 is flat (21.26 vs
21.32 ms, noise); the win is one launch/step instead of chunked
multi_tensor_apply and a place to bolt on future fusions. AdamW bucket
is done; do not revisit.

Pointwise/copy audit (#25) + gather_cast (merged e0cf613). Attribution
of the 7.3% aten bucket on GPU0: the ring batch path was the majority
(index_select gather copy, uint8 reshape copy, uint8->fp32 CL convert,
in-place mul_, autocast bf16 cast at the stem: ~600 MB of traffic per
step at production shape), then eager GELU (bucket-internal, kC's
target), autocast weight casts (bfloat16_copy ~29/step, small), and
cudagraph static input copies from the compiled temporal (direct_copy
~24/step, tens of us; attributed, not attacked). Fix: Triton
gather_cast (slot gather + uint8->fp32 scale -> bf16 channels_last, one
pass, ~100 MB) + scale_cast sibling for preprocess of goal/neg and
non-ring uint8 sources; loader now emits bf16 CL batches directly and
prepare_images no-ops on them. Bit-exact vs refs by construction
(uint8->fp32 mul, one rounding at out dtype); 13 new tests bitwise.
Bench (RTX PRO 6000 Blackwell, bench_gather_cast.py): batch chain
0.2074 -> 0.0181 ms (11.4x), scale_cast 0.0751 -> 0.0193 ms (3.9x).
Post-merge GPU1 profile (a171222): wall 21.2 -> 20.62 ms/step
(self-time 20.26); _gather_cast_kernel sits at 0.102 ms/step, exactly
the 3090 BW roofline (100 MB / 936 GB/s); the convert/mul/index chain
is gone from the kernel dump. GPU0 e2e delta lands with the kC batch
profile (GPU0 occupied by the kC delegate at write time).

## 2026-07-15 - kC conv frontend merged; cudagraph-trees NaN hunt blocked the push

kC (delegated, codex): fused channels-last conv+exact-GELU frontend for the
encoder (stem 3->32 7x7/s2 + block1 32->64 3x3/s2), custom Triton fwd
(stores both y and pre-activation), epilogue-backward (GELU' from pre),
packed stem dgrad, generic dgrad, split-K wgrad into fp32. Guards pin the
Triton path to the exact production shapes/dtypes/CL layouts; everything
else falls back to the pure-torch ref. Acceptance re-run by us before
merge (68c15c1): encoder fwd+bwd at production rows -27.7%, stem leg
1.76x, block1 1.10x vs cudnn+eager-GELU baseline; delegate-reported e2e
7.86 -> 6.97 ms on GPU0 in its worktree.

Posttrain smoke then NaN'd deterministically at posttrain step 2 (grad
nan on encoder.stem.conv.weight), so the merge was held unpushed. Full
bisect (probe scripts, /tmp/nan_probe*.py): the NaN requires the
COMBINATION conv_gelu + FusedAdamW + cudagraph trees (PAN2_TEMPORAL_COMPILE
at reduce-overhead) + a real train_steps pretrain phase + an eval
forward without autocast in the same process before posttrain. Any one
removed: clean. Manual eager stepping in both stages: clean. Anomaly
detection localizes the first nan output to `_ConvGeluBackward` (block1
dx path, propagating up to stem). compute-sanitizer memcheck: zero
invalid accesses (our kernels hygienic). PYTORCH_NO_CUDA_MEMORY_CACHING=1
trips torch's own internal invariant ("storage data ptrs not allocated
in pool... could be a bug in inductor aliasing tracking"), which put
cudagraph-trees pool bookkeeping in the frame.

Two failed "cures", recorded so nobody retries them:
`torch.compiler.cudagraph_mark_step_begin()` at build_state, per
iteration, and before eval forwards did NOT fix it (first single clean
run was a false signal; 2/2 reruns NaN'd), and dynamo.reset variants
moved the nan rather than removing it. Lesson now applied repo-wide:
believe a NaN cure only after 3/3 reruns.

Resolution: PAN2_TEMPORAL_COMPILE_MODE env added; production default
flipped to mode="default" (inductor fusions intact, cudagraphs off).
3/3 identical clean runs of the exact former repro. Cost measured on
GPU0 (RTX PRO 6000 Blackwell, profile_step.py, 60-step wall):
reduce-overhead 6.85 ms/step vs default 7.19 ms/step, i.e. cudagraphs
buys ~0.34 ms of launch-gap elimination; kernel self-time identical
(5.48 vs 5.50 ms). Not worth a landmine that detonates on any
in-process eval or second compiled model; "reduce-overhead" stays
available via env for single-model benchmarking. The build_state /
per-iteration / pre-eval marks were removed again as non-fixes;
CLAUDE.md carries the constraint. smoke.py now hard-fails on
non-finite loss: before this gate, a NaN train run still printed
"smoke ok".

Post-everything GPU0 bucket map (production default mode, pretrain
shape, wall 7.19 ms/step, self 5.48): gemm 1.673 (30.5%), conv_bwd
1.126 (20.6%; includes remaining cudnn dgrad/wgrad calls worth a look),
other 1.050 (19.2%; our Triton kernels mostly land here), adamw 0.442
(8.1%), aten_pointwise 0.342 (6.2%), inductor_reduction 0.331 (6.0%),
gn_gelu 0.255 (4.6%), inductor_pointwise 0.192 (3.5%), conv_fwd 0.050
(0.9%), groupnorm_eager 0.017. Host-side residual 1.71 ms/step. Our
largest single kernels: _conv_gelu_wgrad 0.561 + _conv_gelu_fwd 0.473.
This closes #25's GPU0 end-to-end profile; next-largest honest target
is the gemm/opportunity work queued as kE (#26).

## 2026-07-15 - kE epilogue-fused gemms: honest negative, and round 2 of the NaN

kE (delegated, codex): fuse fc1+bias+exact-GELU and proj/fc2+residual
into Triton matmuls with M=2144 wave-tail tile tuning. Result: NOTHING
landed and the branch was deleted - every candidate lost to the
cuBLAS+pointwise baseline it replaces. Delegate-measured on GPU0
(uncontended): proj+residual addmm 0.87x of existing (0.0153 vs 0.0132
ms), fc2 bias+residual 0.85x, fc1 fused GELU no qualifying result; on
the 3090 every Triton fwd+bwd path also lost (fc1 0.3371 vs 0.4203 ms,
proj 0.0918 vs 0.1300, fc2 0.2788 vs 0.3658; baseline vs fused).
64-row tiles improved the wave tail but not enough to beat cuBLAS at
these bf16 shapes. Conclusion recorded so nobody re-rolls plain
block-matmul Triton at M=2144x{512,1536,2048}: the gemm bucket
(1.673/5.48 ms/step self) stays; remaining levers are
max-autotune-gemm-style experiments or batch-shape decisions, i.e.
config/architecture moves rather than hand kernels.

Round 2 of the NaN (triggered by a kE side observation). The kE
delegate reported baseline @ 497e01e NaNing posttrain on GPU1 under
compile-default. First suspicion was inductor-cache poisoning from
delegate experiments (a fresh cache dir produced one clean GPU1 run).
Disproven: after wiping /tmp/torchinductor_infatoshi aside, GPU1
default-mode NaN'd 3/3 on the repopulating cache and 3/3 more on three
independent fresh cache dirs (one earlier single clean fresh-cache run
was another false signal - the repo rule "believe a cure after 3/3"
applies to controls too). GPU0 meanwhile kept its split behavior on
the fresh cache: default clean, reduce-overhead NaN. GPU1 bisect:
PAN2_FUSED_ADAMW=0 clean, conv_gelu removed clean, both present NaN -
the SAME three-component combination as GPU0 (conv_gelu + FusedAdamW +
inductor compile), and anomaly detection names the same site on both
arches: "_ConvGeluBackward returned nan values in its 0th output"
(block1 dx). Mechanism beyond the triad is OPEN (not cudagraphs on
GPU1, not the cache, not memcheck-visible OOB). Production rules now in
CLAUDE.md: GPU0 runs compile mode=default (verified clean 4/4 across
old and fresh caches; it is also the primary target), GPU1 runs with
PAN2_TEMPORAL_COMPILE=0 until root-caused. The delegate-facing lesson
that DID stick: delegates get cache-hygiene instructions in prompts
(any deterministic-NaN investigation must include a fresh-cache
control), and GPU0 was contended by KernelBench-Hard during kE -
bench attribution above is uncontended windows only.

Campaign state: all delegated kernel lanes closed. Merged stack on
main: gather_cast/scale_cast (#25), fused AdamW (kD), conv_gelu
frontend (kC), compile-mode fix (497e01e). Remaining open items:
whole-model compile at mode=default for the 1.71 ms host-side residual
(loop.py:47 still selects reduce-overhead when cfg.train.compile=true -
do not flip that on as-is), the bs=64 sample-efficiency question with
the user, then data-pipeline work dominates per the SPEC (DATA is the
binding constraint, not kernels).

## 2026-07-15 - the NaN is a conv_gelu dgrad flake; kC Triton path default-off

The story collapsed and re-formed twice today; this is the final form.
A real-data finite check - which no gate ran before, benchmarks never
checked losses - showed the production stack NaNing real pretrain
training at step 3-13 on GPU0 with healthy grad norms right up to the
flip (bs32 and bs64 alike, single model, no eval). Bisect on the
real-data oracle: PAN2_FUSED_ADAMW=0 nan 3/3, temporal compile off
nan, gather family off nan, bias family off nan, conv_gelu off = 60
steps finite. conv_gelu's Triton path alone is necessary and
sufficient; every earlier "triad" result from the synthetic smoke
hunts was allocator-layout flips perturbing a probabilistic bug
(synthetic smoke sees ~15 dgrad calls, p(nan per smoke) only ~14%, so
"clean 3/3" was always weak; the two cache/marking false cures were
the same trap). Rule now enforced: only oracles with hundreds of calls
(60-step real-data runs) or hard gates count as evidence here.

Instrumentation (monkeypatched _ConvGelu.backward, real data): first
nan is an OUTPUT of the block1 (cin=32) generic dgrad kernel at
backward call 112 - dx nan, dweight clean, grad_output/x/weight/pre
all verified finite before and after. Zero-initialized dx still NaNs
3/3, so the kernel computes nan from finite inputs; it is not reading
stale uninitialized bytes. It passes isolated unit tests at production
shapes consistently, which is how the merge shipped it - real-loop
allocator/stream state matters to whatever mis-indexed load or
miscompile this is (both sm_120 and sm_86 reproduce).

Action: PAN2_CONV_GELU_TRITON env, default OFF (ref conv+GELU serves
everywhere; state_dict unaffected; kC's unit tests now vacuously
compare ref vs ref - noted). Cost measured on GPU0: wall 7.19 -> 7.78
ms/step, kernel self 5.48 -> 6.48. Gates on the disabled stack: pytest
73/73, ruff clean, 3x 60-step real-data runs finite at bs32, 2x at
bs=64, 2x smoke, GPU1 60-step real-data + compile-default smoke clean
(GPU1's nan was this same flake, not an arch-specific compile issue).
train_steps now hard-raises on any non-finite loss (the smoke-only
gate did not cover real runs).

Also landed here: bs=64 pretrain default (user-approved; -19%
ms/sample measured in the batch sweep; 2x 60-step real-data finite),
PAN2_MODEL_COMPILE_MODE for the whole-model compile branch (was
hardcoded reduce-overhead; whole-model default-mode compile measured
flat, 7.20 vs 7.14 ms wall GPU0, so cfg compile stays off), probe8
verified 3x clean for whole-model default compile.

Open: dgrad flake root cause (fix lane in a worktree; acceptance =
5x 60-step real-data clean runs + unit tests + benches beating ref).
Recovering the 0.59 ms/step is the prize.

## 2026-07-15 - dgrad flake ROOT-CAUSED and fixed (kF); Triton conv path back ON

Delegate (grok) in worktree kF/conv-dgrad-flake, root cause pinned with
a deterministic counter-test, not vibes:

The generic dgrad kernel's parity-strided filter loop used trip count
ceil(K/STRIDE). For block1 (KH=3, s=2) with kh_start=1 that yields
kh=1,3 - kh=3 is out of bounds for a length-3 filter. dpre loads were
already masked for kh>=KH (0); WEIGHT loads were not. The OOB weight
address reads past the packed channels-last buffer into neighboring
device memory, which under training allocator reuse often contains nan
from earlier grads. Masked-dpre (0) times OOB-weight (nan) is 0*nan =
nan inside tl.dot, poisoning the whole dx tile. Matches every forensic
fact on record: finite verified inputs producing nan (call #112),
~1%/call in real loops only (needs junk adjacent to the weight
allocation), isolated unit tests clean (clean adjacent memory), wgrad
clean (always in-bounds pattern), stem dgrad clean (range(KH) loop),
both sm_120 and sm_86, zeros-init dx no cure (computed, not stale).
num_stages had nothing to do with it.

Fix (3eb3edf, 34 lines): clamp the weight address in-bounds
(kh_safe = min(kh, KH-1), same for kw) and extend the weight load mask
with (kh < KH) & (kw < KW), so the overshoot taps read real in-bounds
words and contribute exactly 0. Bit-exact for all previously-correct
parity cases.

Verification, all re-run by us, not just the delegate:
- deterministic regression test: nan in the 8192 elements after the
  packed weight buffer; old kernel 98304/131072 dx nan, fixed kernel 0
  nan, fixed vs old-with-zero-tail maxdiff 0.0
- triton-vs-ref test asserts _can_use_triton with env forced on (kills
  the ref-vs-ref vacuity noted in the previous entry)
- pytest 77/77 with env off and forced on; ruff clean
- oracle 10/10 60-step real-data runs finite (5x bs32 + 5x bs64,
  PAN2_CONV_GELU_TRITON=1; bs64 doubles conv-bwd calls/step) on top of
  the delegate's own 5/5
- bench GPU1 3090 fwd+bwd N=2080: stem 5.15 vs 7.20 ms ref (1.40x),
  b1 1.94 vs 1.92 (~1.0x, cuDNN parity on Ampere); delegate's GPU0
  bench: stem 3.09x, b1 1.10x

PAN2_CONV_GELU_TRITON default back to ON (=0 forces ref, A/B only).
Recovers the 0.59 ms/step the disable cost (7.78 -> 7.19 ms wall
territory, GPU0, conv-on vs conv-off measured earlier today). The
triad-era comments in temporal.py/loop.py now say plainly that
cudagraph trees were never causal; mode="default" stands on its own
0.34-ms-upside merits. train_steps finite-loss gate stays as the
permanent backstop.

Lesson now codified in the nan-tail regression test: any strided/parity
filter loop with trip count ceil(K/S) must mask EVERY operand of the
dot, not just the data operand - one unmasked operand turns index
overshoot into value nondeterminism that isolated unit tests cannot see.

## 2026-07-16 - Overnight YouTube crawl: 1.10TB / 9,733 videos / ~4,400h in data/crawl

Mission (user, ~00:30): pull ~1TB of Minecraft gameplay from YouTube
into data/, leave 300GB free, autonomous overnight. Delivered.

Result (measured this morning, unique files in raw/ - NOT done-lines):
- 9,733 videos, 1.10TB raw h264 640x360 (30fps and 29.97fps mix),
  meta TSV coverage 9733/9733 = ~4,398 hours on disk
- durations (s): min 600 / med 1440 / p90 2214 / max 10786
- supervisor self-halted on 1TB target at 11:22:15; floor never
  touched (1.33TB free at halt; 1.3TB free now)
- ~4,398h = 39x the 113h VPT corpus; at the measured GPU0 pretrain
  rate of 4.8 h-video/s one epoch is ~15 min

How it ran:
- queue: 43,581 unique ids from 24 hermitcraft/LP channels (full
  catalogs) + 4 ytsearch no-commentary playlists; shuffled, split
  per-worker; archive excludes done ids across restarts
- pilots: yt-dlp per-connection server throttle ~1MB/s REGARDLESS of
  --concurrent-fragments (3 vs 8 identical); aria2c -x 8 -s 8 -k 16M
  = 8.1MB/s per worker - 8x, installed and wired via --downloader
- 01:46 W=40 launch tripped the YouTube IP bot-wall (~80 player-API
  calls/min incl retries); parked fleet with state/STOP, shaped
  relaunch W=12 + --sleep-requests 2 + extractor-retries 1 +
  6-consecutive-bot-wall self-cooling per worker; auto_restart.sh
  probe-gates relaunches (2 consecutive PASS 300s apart + 3-probe
  spread >= 2). Wall lifts after ~11 min cool-down and does not
  escalate across ~30 autonomous wall/cool cycles overnight
- healthy-fleet sustained rate ~730-800Mbps -> 1TB ideal ~3.1h;
  actual ~9.6h through the cool-down cycles
- worker rc=0-with-archive-skip still appends a done line, so
  done-lines (16,067) overstate unique files (9,733) across the ~30
  relaunch restarts. Ground truth = unique files in raw/. Fails
  (8,812 lines) are dominantly filter rejects (shorts <600s,
  is_live), age-gates, members-only, plus wall streaks - content
  composition, not poison. 18,606 queue ids remain unprocessed.

QA:
- ffprobe spot-check 15/15 healthy h264 640x360
- archive-vs-raw comm: 0 archive-only ids (every archived id has its
  file; the fail path rms partials)
- ref transcodes frame-exact vs duration*10 (e.g. 12474/12474);
  0 tc_fail in 773 at check time
- title scan: ~39% non-Minecraft-looking titles (upper bound from a
  crude title regex; whole-channel catalogs include Don't Starve
  etc). Filter at shard build; raw kept.

Reference-view transcode (docs/ingest-codec.md contract: fps=10 ->
lanczos 128x128, x265 medium crf28, GOP 20, audio dropped, frame-count
verified): pool restructured morning-side from 3-wide batch+wait to
12-wide continuous refill (tc_run markers make re-entry safe), ETA ~6h
for the full ~9.7k set, projected ~87GB total (773 done = 6.9GB).

Codec arms (overnight, answer to ingest-codec.md open item): 3 arms x
1000 steps, identical init, train-batch argmax accuracy vs arange:
A(shard) 0.38->0.9975, B(crf28) 0.41->1.0, C(nvenc_cq32) 0.11->0.999
by step 400-1000. Coarse contrastive signal survives crf28; crf28
green-lit for training data. Fine-detail probes still open.

Lessons:
- pgrep/pkill -f matches the invoking zsh -c command line - every
  self-match killed my own wrapper mid-compound (exit 144). Bracket
  trick or kill-by-pid only; watchers must run in their own pgid.
- Whole-channel harvests buy volume and diversity but import
  non-target content; price it into the queue and the QA, or seed
  with playlists instead.
- aria2 range-parallelism beats yt-dlp fragment parallelism 8x under
  YouTube per-connection throttling.

Open/next: preview training path on the new corpus (build_shards.py
needs an --fps flag: decode_mp4 has no fps handling; plan fps=20 ->
existing 20Hz/64px contract, zero pipeline changes; 128px/10fps
Pan-1-aligned layout is a data-layout decision for the user).

## 2026-07-16 (pm) - Custom pack data layout: 39GB corpus, 4.44 h/s E2E

User directive: custom data layout, small, decompress on the fly
superfast, training as fast as possible. Shipped v1 end-to-end and
measured every load-bearing number first.

Engine selection (benches on real refs, box co-running transcodes):
- seek-per-window random access caps everything: PyAV 1.1k fps (rgb
  conversion ~7x the decode), torchcodec cpu 6.6k, torchcodec NVDEC 4.0k
- NVDEC DEAD for v1: rejects width<144 ("not within range from 144 to
  8192" - our 128px and any 64px refs); at 144px every ffmpeg/torchcodec
  path syncs per frame -> 7-12k fps, no 4-stream scaling. v2 = async
  surface queue or nvJPEG, documented in docs/data-layout.md
- sequential ffmpeg: 11-26k fps/instance @128px frame-threaded, ~2.5x
  cheaper/frame @64px (same-video A/B)
- ring math (measured constants): refresh_prob 0.05 -> ~146 fresh
  slots/s x 428 frames = ~54k decoded fps demand. 64px CPU envelope
  holds that with 12 stateless producers -> persistent pipes / annexb /
  GPU decode all buy nothing in v1

Layout (docs/data-layout.md is the contract): ref64/ (64px/10fps x265
crf28 GOP20, EXACT 1:1 frame equality vs ref/) + pack_index.npz (exact
n_frames, crude title-only minecraft flag - channel names excluded,
they flag everything), 8.6MB/h => ~39GB per 4,400h. Loader:
prefer_source="pack", -ss-before-i exact-frame seeks (bit-exact test at
GOP straddles), decode-at-native no scale, native_fps plumbing, exact
max_start. stride exactly once (10fps ingest = the subsample; k=1 ring
and model).

E2E (GPU0 lease, bs64, ctx128@10fps, 1000 steps, 81 episodes / 34.1h):
51-54 ms/step steady = 4.44 h-video/s vs 4.80 shard-path (delta =
co-running encode CPU). errors=0, 5455 fills, ring never starved, loss
1.10 -> 0.010. 7 new tests (index builder 2, loader parity 5), pytest
84/84, ruff clean.

Lessons:
- compute the fill demand before picking a decode engine: the 350k fps
  "requirement" (every step freshly decoded) was 6.5x the real ring
  number; it was driving the design toward a GPU-decode build-out the
  ring makes unnecessary
- NVDEC tiny-frame reality: min-width 144 plus sync-per-frame APIs ->
  hardware decoders are a non-option until someone owns an async queue
- ffprobe-counted frame counts in an index beat act-file probing:
  exact max_start removes the mp4 branch's 4000/5000 guesses

Next: tonight's preview run (rebuild index over the grown ref64 set
~21:00, 100k steps ~90 min on GPU0); topic queue stays: wrong-horizon
negatives, cross-episode goals, fine-detail codec probes.
