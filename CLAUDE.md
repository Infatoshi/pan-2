# pan-2

Open-source goal-conditioned Minecraft agent (MIT). Public canonical:
`github.com/Infatoshi/pan-2`. Working checkout + data: `anvil:~/dev/pan-2`.
Train/eval on anvil GPU0 (RTX PRO 6000 Blackwell sm_120) when free, GPU1
(3090) otherwise. Mac is control plane only.

## What
Single-GPU reproduction of Pantograph's Pan-1 (journal:
pantograph.com/journal/pan-1, alignment notes in SPEC.md). Two stages:
action-free goal pretrain on video (strictly future-frame goals, InfoNCE +
same-episode hard negatives), then action post-train on VPT-style labeled
episodes with a chunked action head. The binding constraint vs Pan-1 is DATA
(113h vs their 500k h video / 2k h demos); prioritize data acquisition over
model churn.

## Docs
- Method/architecture contract: `SPEC.md`
- Journey with measured numbers: `DEVLOG.md`
- Public-facing intro/install/data contract: `README.md`
- This file: agent commands + constraints only

## Layout
- `src/pan2/actions.py` — action layout source of truth (measured; never guess)
- `src/pan2/data/` — `vpt_episodes.py` (mmap), `gpu_pipeline.py` (ring), `synthetic.py`
- `src/pan2/models/` — encoder / temporal / heads / policy / preprocess
- `src/pan2/train/`, `src/pan2/eval/`
- `src/pan2/kernels/` — custom ops home. Contract in its `__init__.py`:
  every op = pure-torch `*_ref` + optimized impl + registry + unit test +
  bench at production shapes. Speed claims must cite bench output.

## GPU
- Prefer GPU0 Blackwell 96GB. Check `nvidia-smi` before long runs (shared box).
- Overnight: follow anvil-shared-gpu / overnight-compute rules if contending.

## Data
- Canonical train data: `/data/pan-2/` (symlink: `~/dev/pan-2/data/vpt`)
  - `episodes/` — 1625 clean img.npy+act.npy pairs (~113h @ 20Hz, 64x64, act dim 25)
  - `shards/` — packed train data (24 shards, 8.16M frames, built 2026-07-15; train scripts read this)
  - `raw/` — matching mp4+jsonl for good stems
  - `meta/` — manifest + cleanup report
- Local artifacts: `~/dev/pan-2/data/{checkpoints,cache,shards}`
- YouTube crawl corpus (gitignored, anvil-local): `~/dev/pan-2/data/crawl/`
  - `raw/` — 9,733 scraped videos, 1.10TB (~4,400h); `ref/` — verified 128px/10fps crf28 reference views
  - `ref64/` — 64px/10fps crf28 refs (1:1 frames vs ref/) — train reads this via `--prefer-source pack`
  - `pack/pack_index.npz` — episode index; rebuild with `scripts/build_pack_index.py` as ref64 grows
  - Layout contract: `docs/data-layout.md`
  - `state/STOP` halts workers, `state/STOP_TC` halts transcode; `bin/` has worker/supervisor/transcode scripts; logs in `logs/`

## Commands
```bash
cd ~/dev/pan-2
uv sync --extra dev
uv run pytest -q
uv run ruff check . --fix
uv run python scripts/smoke.py
uv run python scripts/train_pretrain.py --config configs/pretrain_smoke.yaml
uv run python scripts/train_posttrain.py --config configs/posttrain_smoke.yaml
uv run python scripts/train_pretrain_pipeline.py --config configs/default.yaml --budget-gb 10
uv run python scripts/train_posttrain.py --config configs/posttrain_overfit.yaml
```

## Constraints
- UV only for Python (`uv run`, `uv add`). Never bare pip/python for project deps.
- No emojis, no em dashes in docs/commits.
- Tests mandatory for non-trivial changes; smoke must pass before claiming train path works.
- Public repo: keep README/SPEC free of internal infra; anvil paths belong here only.
- Subsample stride applied exactly once (model, or ring fill + model k=1). Never both.
- Goals are strictly future frames; never reintroduce in-window goals.
- Kernel speed is earned: ref impl + test + bench land in the same change as the kernel.
- conv_gelu dgrad nan flake (2026-07-15): root-caused and fixed same-day (kF, 3eb3edf). The generic dgrad's parity-strided filter loop overshot K when start != 0 (KH=3,s=2,kh_start=1 -> kh=1,3 OOB); dpre loads were masked, weight loads were not, so OOB weight garbage (often nan under allocator reuse) entered tl.dot as 0*nan -> nan dx from finite inputs at ~1% of calls in real loops (isolated tests clean). Fix: clamp weight address in-bounds + mask kh/kw >= K; regression test poisons the weight tail deterministically. Verified 10/10 60-step real-data runs (bs32+bs64, delegate + independent re-runs). Triton path default ON again; `PAN2_CONV_GELU_TRITON=0` forces the ref (A/B/debug only). All earlier "conv_gelu+FusedAdamW+compile triad" / cudagraph observations were allocator-layout artifacts of this one bug; the mode="default" compile guidance stands on its own (0.34 ms upside, not nan safety). `train_steps` still hard-raises on non-finite loss. Perf claims must state whether the conv path was on or off.
- Do not start multi-day pretrain until data layout is agreed and free space verified on `/` and `/data`.
