# pan-2

Single-GPU reproduction of Pantograph's Pan-1 method
(https://pantograph.com/journal/pan-1) for goal-conditioned Minecraft agents. Two stages:

1. **Stage A: action-free goal pretrain.** A causal transformer over CNN frame
   tokens learns, from observation-only video, to match a context window to a
   frame that lies strictly in its future (InfoNCE, hindsight goals with a
   1..15s horizon, plus a same-episode hard negative per row so the task
   cannot be solved by scene matching alone).
2. **Stage B: action post-train.** Attach a chunked action head
   (`pi(a[t:t+H] | history, goal)`) and behavior-clone VPT-style labeled
   episodes, keeping goal conditioning via hindsight goals.

Stage C (online hindsight relabeling from rollouts) is planned, not required.

Status: alpha research code. The pretrain/posttrain loops, data pipeline, and
overfit sanity checks run; there is no Minecraft env integration yet.

Honest gap to Pan-1 (verified against the journal): Pan-1 used ~500k h of
action-free video, ~2k h of contractor trajectories, and a 104-env grader
suite. This repo currently has 113 h / 113 h / 0 envs and a much smaller
model. Closing the data gap is the active work; the training machinery here
is the vehicle, not the result.

## Why it exists

VPT showed behavior cloning on contractor video works; goal-conditioned
pretraining (HER / contrastive GCRL style) gives the policy a task signal from
unlabeled video so "watching" transfers to "doing". pan-2 packs that pipeline
into a single-GPU box with a data path that does not become the bottleneck.

## Install

```bash
uv sync --extra dev   # torch 2.13 + cu13 on our box; any recent CUDA torch works
```

UV manages the environment (`uv run` everywhere, never bare pip).

## Data

VPT-style episodes as paired numpy files in one directory:

```
episodes/
  <stem>.img.npy   # (T, 64, 64, 3) uint8 frames at 20 Hz
  <stem>.act.npy   # (T, 25) float32 actions
```

Action layout (25 dims, measured against raw VPT jsonl, see
`src/pan2/actions.py`): cols 0-22 binary buttons
(w/a/s/d, space, lshift, lctrl, e, esc, mouse.0/1; hotbar columns dead in our
data), cols 23-24 camera dx/dy quantized to 0.1 steps in [-1, 1].

Optional raw mp4+jsonl (same stems, sibling dir) enables the ffmpeg producer
in the GPU pipeline; the npy path is the default and is what we benchmark.

Point `train.data_dir` at your episodes dir (config or defaults assume
`/data/pan-2/episodes`).

## Quickstart

```bash
uv run pytest -q
uv run python scripts/smoke.py                                        # synthetic, both stages
uv run python scripts/train_pretrain.py --config configs/default.yaml # Stage A (mmap loader)
uv run python scripts/train_pretrain_pipeline.py --budget-gb 10       # Stage A (GPU ring loader, fast path)
uv run python scripts/train_posttrain.py --config configs/posttrain_overfit.yaml  # 1-episode overfit sanity
uv run python scripts/bench_pretrain.py                               # throughput phases
```

Reference throughput (27M params, T=128, 65 tokens/clip, bs=32, bf16, RTX
3090): 31 ms/step on the GPU ring loader path (~67k frame-tokens/s), 40
ms/step on the mmap loader path.

## Layout

```
src/pan2/
  actions.py          # measured VPT action layout (single source of truth)
  config.py           # dataclass configs, yaml loading
  data/
    vpt_episodes.py   # mmap episode dataset (future goals, hard negatives)
    gpu_pipeline.py   # GPU-resident clip ring with async producers
    synthetic.py      # plumbing fixtures (shape-matched, unlearnable by design)
  models/
    encoder.py        # CNN frame encoder, 1 token/frame
    temporal.py       # causal transformer (SDPA); hook for other backbones
    heads.py          # goal value head (contrastive), action chunk head
    policy.py         # PanPolicy: single temporal subsampler owns stride
    preprocess.py     # uint8 -> float, GPU resize
  train/              # loop, losses, speed knobs
  eval/               # contrastive metrics
  kernels/            # CUSTOM OPS LIVE HERE (contract + registry, see below)
configs/              # yaml: smoke / default / fast / overfit
scripts/              # train entries, benches, profilers, smoke
tests/                # pytest, incl. future-goal regression tests
SPEC.md               # architecture and method contract
DEVLOG.md             # honest journey, numbers included
```

## Custom kernels

`pan2/kernels/` is where hand-written ops (Triton/CUDA/compile) land. Rule set
(short version): every op ships a pure-torch `*_ref`, the optimized impl, a
registry entry, a unit test comparing the two, and a bench at production
shapes. Speed claims cite bench output. Full contract:
`src/pan2/kernels/__init__.py`.

## Method notes / pitfalls we already hit

- In-window goals collapse contrastive pretraining to duplicate detection.
  Goals must be strictly future frames.
- Subsample exactly once. Data-side and model-side stride both applying used
  to silently encode 3 tokens per clip.
- If your contrastive accuracy saturates in a few hundred steps, your
  negatives are too easy; ours were, until same-episode negatives landed.
- 64px native beats upsampled 128px. Trust the data's resolution.

## Docs

- `SPEC.md` method and architecture contract
- `DEVLOG.md` dated engineering log with measured numbers
- `AGENTS.md` / `CLAUDE.md` agent-facing commands and constraints

## References

- Pan-1 journal (the method this repo reproduces):
  https://pantograph.com/journal/pan-1
- VPT: Video PreTraining (Baker et al., 2022), data format + Minecraft BC stack
- HER (Andrychowicz et al., 2017) and contrastive GCRL for hindsight goals
- ACT / BAKU for action chunking

## License

MIT. See `LICENSE`.
