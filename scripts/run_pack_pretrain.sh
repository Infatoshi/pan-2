#!/usr/bin/env bash
# Freeze-tolerant pack pretrain: rebuild the index once, then run the
# trainer with --resume auto in a retry loop so a process crash (or a
# post-freeze relaunch of this script) resumes from the newest checkpoint
# instead of restarting. Idempotent: rerunning after completion exits fast.
# Crash context: anvil silent-freeze forensics 2026-07-17 (DEVLOG).
# Usage: scripts/run_pack_pretrain.sh [max_steps=1000000] [producers=12] [config]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MAX_STEPS="${1:-1000000}"
PRODUCERS="${2:-12}"
CFG="${3:-configs/pretrain_pack.yaml}"
IDX="data/crawl/pack/pack_index.npz"
RETRIES=50

# cudagraph trees on the temporal stack: re-measured 2026-07-17 under live
# producer GIL contention at bs64/T128 - worth ~8 ms/step (38.4 -> 30),
# not the 0.34 ms of the quiet bs32 bench. Loss parity validated 2.5k steps.
export PAN2_TEMPORAL_COMPILE_MODE="${PAN2_TEMPORAL_COMPILE_MODE:-reduce-overhead}"

uv run python scripts/build_pack_index.py \
	--ref64-dir data/crawl/ref64 --out "$IDX" \
	--meta data/crawl/state/meta/list.tsv --workers 12

LOG="data/crawl/logs/pack_pretrain_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data/crawl/logs
echo "logging to $LOG (appending across retries)"

for attempt in $(seq 1 "$RETRIES"); do
	echo "attempt $attempt/$RETRIES $(date -Is)" | tee -a "$LOG"
	uv run python -u scripts/train_pretrain_pipeline.py \
		--config "$CFG" \
		--prefer-source pack --pack-index "$IDX" \
		--producers "$PRODUCERS" --max-steps "$MAX_STEPS" \
		--resume auto 2>&1 |
		while IFS= read -r line; do echo "$(date +%s) $line"; done |
		tee -a "$LOG"
	rc=${PIPESTATUS[0]}
	if [ "$rc" -eq 0 ]; then
		echo "pretrain complete rc=0 $(date -Is)" | tee -a "$LOG"
		exit 0
	fi
	echo "trainer exited rc=$rc, resuming in 15s" | tee -a "$LOG"
	sleep 15
done
echo "gave up after $RETRIES attempts" | tee -a "$LOG"
exit 1
