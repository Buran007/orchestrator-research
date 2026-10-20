#!/usr/bin/env bash
# Сервисы для обучения и оценки (docs/deployment.md):
#   frozen     vLLM chat-сервер для замороженных субагентов и судьи (URL: subagents.base_url). Поднимается ТОЛЬКО
#              при локальном URL (127.0.0.1 или localhost); удалённый API — сообщение и пропуск. Локальный сервер
#              отдаёт ОДНУ модель, поэтому все значения subagents.models должны совпадать (иначе start падает).
#              Доля карты: FROZEN_GPU_UTIL (по умолчанию 0.10).
#   embedding  vLLM pooling-сервер для эмбеддера запросов (URL: embedding.base_url). Поднимается ТОЛЬКО при
#              локальном URL. Удалённый эмбеддер (например OpenRouter) — сообщение и пропуск: ни запуска,
#              ни остановки, ни health-проверки.
#              Доля карты: EMBED_GPU_UTIL (по умолчанию 0.05).
#   retrieval  orchestrator-research serve-retrieval (CPU), FAISS-индекс грузится один раз (URL: retrieval.server_url)
# start: идемпотентен (живой pid не трогаем); pid-файлы и логи в artifacts/services/; ждёт health
#        (WAIT_SECONDS, по умолчанию 900). vLLM-серверы стартуют строго по очереди: каждый профилирует память
#        при загрузке, второй нельзя запускать, пока грузится первый.
# stop:  SIGTERM, затем SIGKILL. Гасит только процессы с pid-файлом в artifacts/services/; agl-server и ray
#        не трогает.
# status: только чтение; pid, health, память GPU, свежие каталоги прогонов.
# Env: WAIT_SECONDS, FROZEN_GPU_UTIL, FROZEN_MAX_SEQS, FROZEN_KV_BYTES, EMBED_GPU_UTIL, EMBED_KV_BYTES.
# Usage (repo root): bash scripts/services.sh {start|stop|status} CONFIG
set -euo pipefail
source "$(dirname "$0")/lib.sh"

CMD="${1:-}"
CONFIG="${2:-}"
case "$CMD" in start | stop | status) ;; *) CONFIG="" ;; esac
[ -n "$CONFIG" ] || { echo "usage: bash scripts/services.sh {start|stop|status} CONFIG" >&2; exit 2; }

SVC_DIR="artifacts/services"
WAIT_SECONDS="${WAIT_SECONDS:-900}"
FROZEN_GPU_UTIL="${FROZEN_GPU_UTIL:-0.10}"
EMBED_GPU_UTIL="${EMBED_GPU_UTIL:-0.05}"
SUB_URL="$(cfg_get "$CONFIG" subagents.base_url)"
SUB_MODELS="$("$PY" -c 'import sys
from orchestrator_research.config import load_config
print("\n".join(load_config(sys.argv[1]).subagents.models.values()))' "$CONFIG")"
EMB_MODEL="$(cfg_get "$CONFIG" embedding.model)"
EMB_URL="$(cfg_get "$CONFIG" embedding.base_url)"
RET_URL="$(cfg_get "$CONFIG" retrieval.server_url)"

# launch NAME COMMAND...: nohup в фоне, pid в $SVC_DIR/NAME.pid; если процесс уже жив — ничего не делает.
launch() {
    local name="$1"; shift
    local pid_file="$SVC_DIR/$name.pid" log="$SVC_DIR/$name.log"
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "$name already running (pid $(cat "$pid_file"))"
        return 0
    fi
    nohup "$@" >>"$log" 2>&1 </dev/null &
    echo $! >"$pid_file"
    echo "started $name (pid $!), log $log"
}

# check_frozen_model: FROZEN_MODEL = модель первого алиаса. Выход из скрипта, если алиасы ссылаются на разные
# модели: локальный vLLM-сервер отдаёт одну.
check_frozen_model() {
    local distinct count
    distinct="$(printf '%s\n' "$SUB_MODELS" | sort -u)"
    count="$(grep -c . <<<"$distinct" || true)"
    if [ "$count" -ne 1 ]; then
        echo "error: subagents.base_url is local ($SUB_URL), so one vLLM server serves it, but" >&2
        echo "       subagents.models names $count different models:" >&2
        sed 's/^/         /' <<<"$distinct" >&2
        echo "fix: set every alias in [subagents.models] to the same model id, or point subagents.base_url at a remote API." >&2
        exit 1
    fi
    FROZEN_MODEL="$(head -n 1 <<<"$SUB_MODELS")"
}

start() {
    mkdir -p "$SVC_DIR"
    if is_local_url "$SUB_URL"; then
        check_frozen_model
        if is_local_url "$EMB_URL" && [ "$(url_port "$SUB_URL")" = "$(url_port "$EMB_URL")" ]; then
            echo "error: subagents.base_url and embedding.base_url are both local on port $(url_port "$SUB_URL")" >&2
            exit 1
        fi
        # --gpu-memory-utilization — доля ВСЕЙ карты (0.10 x 80 GB = 8 GB); если столько не свободно, vLLM не стартует.
        launch frozen .venv/bin/vllm serve "$FROZEN_MODEL" \
            --host "$(url_host "$SUB_URL")" --port "$(url_port "$SUB_URL")" --served-model-name "$FROZEN_MODEL" \
            --gpu-memory-utilization "$FROZEN_GPU_UTIL" --max-model-len 8192 --max-num-seqs "${FROZEN_MAX_SEQS:-64}" --max-num-batched-tokens 2048 \
            --kv-cache-memory-bytes "${FROZEN_KV_BYTES:-3000000000}" \
            --enable-auto-tool-choice --tool-call-parser qwen3_coder --dtype bfloat16
        wait_http "$SUB_URL/models" "$WAIT_SECONDS" "$(cat "$SVC_DIR/frozen.pid")" "$SVC_DIR/frozen.log"
    else
        echo "subagents: remote ($SUB_URL)"
    fi
    if is_local_url "$EMB_URL"; then
        # --gpu-memory-utilization — доля ВСЕЙ карты (0.05 x 80 GB = 4 GB); если столько не свободно, vLLM не стартует.
        launch embedding .venv/bin/vllm serve "$EMB_MODEL" --runner pooling \
            --host 127.0.0.1 --port "$(url_port "$EMB_URL")" --served-model-name "$EMB_MODEL" \
            --gpu-memory-utilization "$EMBED_GPU_UTIL" --max-model-len 512 --max-num-batched-tokens 2048 --kv-cache-memory-bytes "${EMBED_KV_BYTES:-1000000000}"  # только короткие запросы
        # KV-кэш задан в байтах явно: на общей GPU чужие аллокации во время профилирования памяти vLLM
        # засчитываются нам, и одной --gpu-memory-utilization верить нельзя.
        wait_http "$EMB_URL/models" "$WAIT_SECONDS" "$(cat "$SVC_DIR/embedding.pid")" "$SVC_DIR/embedding.log"
    else
        echo "embedder: remote ($EMB_URL)"
    fi
    launch retrieval .venv/bin/orchestrator-research serve-retrieval --config "$CONFIG" \
        --host "$(url_host "$RET_URL")" --port "$(url_port "$RET_URL")"
    wait_http "$RET_URL/healthz" "$WAIT_SECONDS" "$(cat "$SVC_DIR/retrieval.pid")" "$SVC_DIR/retrieval.log"
    echo "services are ready; check with: bash scripts/services.sh status $CONFIG"
}

stop() {
    # pid-файл гасим, даже если конфиг теперь указывает на удалённый сервер: старый процесс всё равно наш.
    if [ -f "$SVC_DIR/frozen.pid" ]; then
        stop_pid "$SVC_DIR/frozen.pid"
    elif is_local_url "$SUB_URL"; then
        echo "frozen: no pid file"
    else
        echo "subagents: remote ($SUB_URL), nothing to stop"
    fi
    if [ -f "$SVC_DIR/embedding.pid" ]; then
        stop_pid "$SVC_DIR/embedding.pid"
    elif is_local_url "$EMB_URL"; then
        echo "embedding: no pid file"
    else
        echo "embedder: remote ($EMB_URL), nothing to stop"
    fi
    stop_pid "$SVC_DIR/retrieval.pid"
}

status() {
    echo "== services"
    local name pid_file
    for name in frozen embedding retrieval; do
        if [ "$name" = frozen ] && ! is_local_url "$SUB_URL"; then
            echo "subagents: remote ($SUB_URL), not managed here"
            continue
        fi
        if [ "$name" = embedding ] && ! is_local_url "$EMB_URL"; then
            echo "embedder: remote ($EMB_URL), not managed here"
            continue
        fi
        pid_file="$SVC_DIR/$name.pid"
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo "$name: running (pid $(cat "$pid_file"))"
        else
            echo "$name: not running"
        fi
    done
    local urls=("$RET_URL/healthz")
    if is_local_url "$SUB_URL"; then
        urls+=("$SUB_URL/models")
    else
        echo "health skipped: subagents are remote ($SUB_URL)"
    fi
    if is_local_url "$EMB_URL"; then
        urls+=("$EMB_URL/models")
    else
        echo "health skipped: embedder is remote ($EMB_URL)"
    fi
    local url
    for url in "${urls[@]}"; do
        if curl -sf -m 5 "$url" >/dev/null 2>&1; then echo "health ok: $url"; else echo "health FAIL: $url"; fi
    done
    echo "== GPU"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
    else
        echo "nvidia-smi not found"
    fi
    echo "== latest runs (newest first)"
    ls -dt artifacts/runs/*/* 2>/dev/null | head -n 5 || true
}

"$CMD"
