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

### Stage A — action-free goal pretrain
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
/data/pan-2/shards/      # train-ready (to create)
~/dev/pan-2/data/        # local experiment artifacts
```

v0 training can read `/data/pan-2/episodes` for post-train and synthetic tensors for unit tests. Full scrape/latent pipeline is data work (user freeing space on the side).

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

- Pantograph Pan-1 journal (method)
- VPT (data + Minecraft BC stack)
- HER / CRL (hindsight + contrastive GCRL)
- BAKU / ACT (action chunking)
