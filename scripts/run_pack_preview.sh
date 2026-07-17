#!/usr/bin/env bash
# Rebuild the pack index over the current ref64 corpus and run the pack
# preview pretrain end-to-end. Layout contract: docs/data-layout.md.
# Assumes a visible CUDA GPU; on anvil, wrap with an overnight-compute lease.
# Usage: scripts/run_pack_preview.sh [max_steps=100000] [producers=12]
set -euo pipefail
cd "$(dirname "$0")/.."

MAX_STEPS="${1:-100000}"
PRODUCERS="${2:-12}"
IDX="data/crawl/pack/pack_index.npz"

uv run python scripts/build_pack_index.py \
    --ref64-dir data/crawl/ref64 --out "$IDX" \
    --meta data/crawl/state/meta/list.tsv --workers 12

LOG="data/crawl/logs/pack_preview_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data/crawl/logs
echo "logging to $LOG"
uv run python -u scripts/train_pretrain_pipeline.py \
    --config configs/pretrain_pack_preview.yaml \
    --prefer-source pack --pack-index "$IDX" \
    --producers "$PRODUCERS" --max-steps "$MAX_STEPS" 2>&1 \
    | while IFS= read -r line; do echo "$(date +%s) $line"; done | tee "$LOG"
