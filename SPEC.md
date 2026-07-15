# SPEC: pan-2

Goal-conditioned visual Minecraft agent trained on a single anvil GPU, method-compatible with Pantograph Pan-1, optimized for single-box wall-clock.

## Problem

Learn a policy that, given recent first-person frames and a goal image, produces mouse/keyboard actions that move the visual state toward the goal. Pretrain goal-directedness from observation-only video (hindsight goals); post-train actions on labeled demos; optional online hindsight later.

## Non-goals (v0)

- Matching Pan-4B on their private 104-env suite
- Multi-node training
- Full generative world model as the policy
- MoE

## Method (three stages)

### Stage A — action-free goal pretrain (v0)
Inputs: frame sequence `f[0:T]`, goal frame `g = f[T - 1 + d]` with horizon
`d ~ U(min_h, max_h)` native frames strictly AFTER the context window
(default 20..300 = 1..15s at 20Hz). The goal must never be a context frame:
an in-window goal collapses the task to duplicate detection.

Learn:
- frame encoder `E`
- temporal backbone `B` over tokens from `E(f)` + goal token `E(g)`
- value / successor head `V(s, g)` (probability / energy that g is achieved)
- optional latent next-frame head for dynamics regularization

Objective family (v0 default): contrastive / InfoNCE over (context, true future goal). Negatives: other goals in-batch (cross-episode, easy) PLUS one same-episode distractor past the goal window per row (hard; scene statistics persist within an episode, so this column defeats scene-ID matching). Toggle: `train.hard_negatives`. Swap-friendly to energy or likelihood later.

### Stage A v1 — Pan-1 alignment (planned, explicitly non-backward-guaranteed)
Verified against https://pantograph.com/journal/pan-1 (2026-07-15):

- Pan-1 definition of V(s, g): "roughly the probability that a goal frame is
  achieved in the agent's FUTURE" — so v0's strictly-future goal extraction
  is Pan-aligned in MECHANISM. Our goal-source data, however, is
  self-centred (we are always the acting agent); Pan's pretraining corpus is
  500k h of arbitrary gameplay, so many (context, goal) pairs cross
  policies/viewpoints, which is what forces the generic value function.
  Cheapest v1 proxy: cross-episode goal sampling, p(other-episode goal) > 0.
- Pan-1 ALSO trains "the next-frame distribution of a goal-conditioned
  policy" — a second objective our v0 lacks entirely. v1 adds a next-frame
  distribution head.
- Pan-1 samples goals from within the 300-frame window. That reads as
  training the value function on full clips/rollouts with goals treated as
  achieved inside the window (their rollouts are 30s, the grader checks at
  any point — achieved means achieved). We keep our future-goal extraction;
  the window-sampling subtlety is about their training data being rollouts,
  which Stage C will revisit.
- Pan-1: 128x128 @ 10FPS, 300-frame context (~30s), 20Hz actions. Our 64px
  comes from the VPT contractor corpus being 64px; the 128px Pan setting is a
  data-scale consequence, not principled upsampling. Our fps/geometry
  (10fps tokens, T=128, 12.8s windows) is Pan-shaped.

### Stage B — action post-train
Same backbone; attach chunked action head `pi(a[t:t+H] | history, g)`.
Train on VPT contractor-style labeled episodes (`img.npy` + `act.npy` under `/data/pan-2/episodes`).
Preserve goal conditioning (sample hindsight goals inside labeled clips).

### Stage C — online hindsight (later)
Env rollouts + hindsight relabel + mix with offline shards. Not required for v0 smoke.

## Architecture (v0)

```
frames [B,T,3,H,W] 128x128
  -> FrameEncoder (CNN) -> [B,T,D]   # 1 token per frame default
goal  [B,3,H,W]
  -> same FrameEncoder -> [B,D]
  -> TemporalBackbone (causal Transformer)
  -> GoalValueHead
  -> ActionChunkHead (post-train only)
```

### Defaults

| Knob | v0 value | Notes |
|------|----------|-------|
| resolution | 64x64 | native data res; 128 was upsampling and buys nothing |
| fps | 10 | = 20Hz native / frame_subsample 2 (single subsampler, see below) |
| context T | 32 smoke / 128 default / 300 target | full-rate frames per window |
| tokens/clip | 65 at T=128, k=2 | grows with T, not with fps sleight of hand |
| tokens/frame | 1 | raise later if needed |
| d_model | 512 | scale 256 then up |
| layers | 8 | |
| action chunk H | 10 | native-rate actions from last context frame |
| param target | ~50-200M first | scale after pipe is hot |

Temporal subsampling happens in exactly one place: in `PanPolicy` for the
mmap dataset path, or at ring fill time for the GPU pipeline path (which then
sets model subsample to 1). Double-applying the stride was a real bug.

### Sequence backbone
Default: causal Transformer (SDPA/Flash). Hook point for Mamba/RWKV later without rewriting data or heads. For T<=300 and 1 tok/frame, transformer is not the bottleneck; encoder + IO are.

### Action space (measured from data 2026-07-15, see `pan2/actions.py`)
act.npy is dim 25: cols 0-22 binary buttons, cols 23-24 camera dx/dy quantized
to 0.1 steps in [-1, 1]. Column-to-key mapping recovered by correlating with
raw jsonl (w/a/s/d, space, lshift, lctrl, e, escape, mouse.0/1 plus 12 dead
columns, presumed hotbar). Chunked over H. Loss: BCE on buttons + MSE on
camera (camera CE over 21 bins is a later option).

## Data layout

```
/data/pan-2/raw/           # source mp4+jsonl (existing)
/data/pan-2/episodes/      # source img.npy + act.npy (existing)
/data/pan-2/shards/        # train-ready packed shards (built)
~/dev/pan-2/data/          # local experiment artifacts
```

### Shard format (`pan2/data/shards.py`, version 1)

Single ingest format for both stages; written by
`scripts/build_shards.py` (from `episodes/` npy pairs, or from `raw/` mp4
with optional recode to a new image size).

```
shards/
  manifest.jsonl          # header row + one segment row per episode
  shard-00000.frames.npy  # uint8 (T, H, W, 3), npy memmap
  shard-00000.act.npy     # float32 (T, 25), same frame offsets
```

Episodes never straddle shards; act rows share frame offsets so one index
serves pretrain and posttrain. `pan2.data.build.episode_dataset(cfg)` picks
ShardDataset when `data_dir/manifest.jsonl` exists, else VPTEpisodeDataset;
both share the windowing contract in `pan2/data/windowing.py` (future goal,
hard negative, action chunk). The GPU ring loader (`prefer_source: shard`,
default `auto` prefers shards when present) fills clips straight from shard
mmaps. 64px build: 24 shards, ~8.16M frames, ~113 h.

Training code reads shards for both stages; the per-episode npy dir stays as
rebuild source. Full scrape/latent pipeline is data work (user freeing space on the side).

### Data acquisition (the gap that matters)

Verified Pan-1 numbers (journal, 2026-07-15): 500k h action-free video,
2k h contractor trajectories, 104-env eval suite. Ours: 113 h / 113 h / 0.
That is ~0.02% and ~5.7% respectively. Model work ahead of the data gap is
premature. Order:

1. Action-free video at scale, 10fps, and NOT self-centred only: Minecraft
   gameplay at 10fps+ with goal-frame extractability (public YouTube
   gameplay in the VPT idm style is the realistic single-box source).
2. Contractor trajectories: replay VPT's acquisition on top of our jsonl
   format (our raw/ and episodes/ are a working template) to grow past 113 h.
3. Env suite: deferred until pretrain shows real retrieval fails on goals
   that need new behavior (not scene ID).

## Training loops

- `scripts/train_pretrain.py` — Stage A
- `scripts/train_posttrain.py` — Stage B (load pretrain ckpt)
- `scripts/smoke.py` — synthetic forward + train steps both stages

Precision: bf16 autocast on Blackwell. Grad clip. AdamW.

## Kernel contract

Custom ops live in `pan2/kernels/`. Every optimized op lands together with a
pure-torch `*_ref`, a registry entry (`pan2.kernels.get`), a unit test
comparing the two, and a bench at production shapes. Speed claims cite bench
output; architecture never blocks on unmeasured microbench.

## Eval (v0)

Offline metrics first:
- goal retrieval accuracy (argmax contrastive)
- action BCE / mouse MSE on held-out contractor
- rollout suite later

## Success criteria

1. `uv run pytest` and `uv run python scripts/smoke.py` pass on GPU0
2. Post-train can overfit a tiny VPT episode subset (sanity)
3. Pretrain contrastive accuracy >> chance on REAL held-out clips
   (synthetic fixtures validate shapes/plumbing only; random-noise frames
   cannot measure retrieval quality by construction)
4. Clear path to scale T, width, and data without API breaks

## References

- Pantograph Pan-1 journal (method): https://pantograph.com/journal/pan-1
  (verified 2026-07-15; v1 alignment notes under Stage A)
- VPT (data + Minecraft BC stack)
- HER / CRL (hindsight + contrastive GCRL)
- BAKU / ACT (action chunking)
