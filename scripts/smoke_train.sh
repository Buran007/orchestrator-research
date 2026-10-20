#!/usr/bin/env bash
# Смоук-прогон: train.sh с --smoke (2 шага, батч 2, 4 строки на сплит) и сэмплер памяти GPU.
# Сэмплер каждые 2 с дописывает nvidia-smi memory.used (MiB) в artifacts/smoke/<run_id>.gpu.csv, гасится на выходе.
# В конце — пик памяти и чеклист по логу обучения.
#
# Usage (from the repo root): bash scripts/smoke_train.sh CONFIG
set -euo pipefail
source "$(dirname "$0")/lib.sh"

CONFIG="${1:?usage: bash scripts/smoke_train.sh CONFIG}"
RUN_ID="smoke-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p artifacts/smoke
GPU_LOG="artifacts/smoke/$RUN_ID.gpu.csv"

(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
        sleep 2
    done
) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT INT TERM

bash scripts/train.sh "$CONFIG" --smoke --run-id "$RUN_ID"
kill "$SAMPLER" 2>/dev/null || true

PEAK_MIB="$(awk 'BEGIN { m = 0 } { if ($1 + 0 > m) m = $1 + 0 } END { print m }' "$GPU_LOG" 2>/dev/null || echo 0)"
echo "peak GPU memory during the smoke run: ${PEAK_MIB} MiB (about $((PEAK_MIB / 1024)) GiB); target < 30 GiB"

LOG="$(ls artifacts/runs/*/"$RUN_ID".train.log 2>/dev/null | head -n 1 || true)"
echo
echo "checklist for ${LOG:-the train log}:"
echo "  [ ] actor/k25_in_band_frac is present and > 0"
echo "  [ ] some rollouts have non-zero reward (reward breakdown in the episode files)"
echo "  [ ] training/n_unmerged_rollouts is printed (> 0 is expected with the Qwen3.5 template, ADR 0005)"
echo "  [ ] peak GPU memory < 30 GiB (docs/deployment.md gives ~27 GiB as an unmeasured estimate)"
echo "  [ ] no traceback in the train log; no error lines in the agl-controller log"
if [ -n "$LOG" ]; then
    echo
    echo "last matching metric lines:"
    grep -E "k25_in_band_frac|n_unmerged_rollouts" "$LOG" | tail -n 6 || echo "  (no matching lines; the run may have failed)"
fi
