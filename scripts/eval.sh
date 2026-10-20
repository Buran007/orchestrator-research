#!/usr/bin/env bash
# Оценка базовой модели (CHECKPOINT_DIR=base) или одного шага verl на одном сплите.
# Шаг сначала мержится: <CKPT>/actor (FSDP-шарды) -> <CKPT>/hf через verl.model_merger; пропускается, если есть
# <CKPT>/hf/config.json (команда сверена с verl v0.8.0). Затем vLLM отдаёт модель как orchestrator.eval_model
# на порту orchestrator.eval_base_url, идёт orchestrator-research eval, на выходе vLLM гасится (trap).
# Нужен services.sh start (retrieval).
# SPLIT: val_test (held-out, для отчётов) | val_full (пересекается с train; как held-out не репортить).
# Env: TOOL_CALL_PARSER (по умолчанию qwen3_coder), ключи из конфига. Имя парсера сверять с `vllm serve --help`
#      для своей версии vLLM: в доках Qwen3-Coder указан qwen3_xml.
# Usage (repo root): bash scripts/eval.sh CONFIG CHECKPOINT_DIR|base SPLIT RUN_ID
# xtrace выключен даже при `bash -x`: ключи из env не должны попасть в вывод.
set +x
set -euo pipefail
source "$(dirname "$0")/lib.sh"

usage() { echo "usage: bash scripts/eval.sh CONFIG CHECKPOINT_DIR|base SPLIT RUN_ID" >&2; exit 2; }
[ "$#" -eq 4 ] || usage
CONFIG="$1"
CKPT="$2"
SPLIT="$3"
RUN_ID="$4"
case "$SPLIT" in val_test | val_full | train) ;; *) usage ;; esac
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

MODEL_PATH="$(cfg_get "$CONFIG" orchestrator.model_path)"
SERVED_NAME="$(cfg_get "$CONFIG" orchestrator.eval_model)"
EVAL_BASE="$(cfg_get "$CONFIG" orchestrator.eval_base_url)"
EVAL_BASE="${EVAL_BASE%/}"
MAX_CONTEXT="$(cfg_get "$CONFIG" training.max_context_tokens)"
RET_URL="$(cfg_get "$CONFIG" retrieval.server_url)"
RET_URL="${RET_URL%/}"
require_env "$(cfg_get "$CONFIG" subagents.api_key_env)" "$(cfg_get "$CONFIG" judge.api_key_env)"

curl -sf -m 10 "$RET_URL/healthz" >/dev/null || {
    echo "error: retrieval server not answering at $RET_URL; run scripts/services.sh start $CONFIG" >&2
    exit 1
}

if [ "$CKPT" = "base" ]; then
    MODEL_DIR="$MODEL_PATH"
else
    CKPT="${CKPT%/}"
    if [ -f "$CKPT/hf/config.json" ]; then
        echo "merged model already at $CKPT/hf; skipping merge"
    else
        .venv/bin/python -m verl.model_merger merge \
            --backend fsdp --local_dir "$CKPT/actor" --target_dir "$CKPT/hf"
    fi
    MODEL_DIR="$CKPT/hf"
fi

VLLM_LOG="artifacts/services/eval-$RUN_ID.vllm.log"
mkdir -p "$(dirname "$VLLM_LOG")"
.venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$(url_port "$EVAL_BASE")" \
    --gpu-memory-utilization "$(cfg_get "$CONFIG" training.rollout_gpu_memory_utilization)" --max-model-len "$MAX_CONTEXT" \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER" \
    >"$VLLM_LOG" 2>&1 </dev/null &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM
wait_http "$EVAL_BASE/models" 900 "$VLLM_PID" "$VLLM_LOG"

.venv/bin/orchestrator-research eval --config "$CONFIG" --split "$SPLIT" --run-id "$RUN_ID" \
    --checkpoint-url "$EVAL_BASE"
echo "eval finished; report with: .venv/bin/orchestrator-research report artifacts/runs/<experiment>/$RUN_ID"
