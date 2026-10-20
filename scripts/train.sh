#!/usr/bin/env bash
# Обучение оркестратора: agl-server (API-шлюз) + agl-controller (local runner) + тренер verl,
# по образцу agentlightning v1.0.2 examples/search_r1/run.sh. Сначала bootstrap_server.sh и services.sh start.
# Порядок: ключи субагентов и судьи заданы -> doctor (FAIL прерывает) -> agl-server, ждём health ->
# agl-controller -> orchestrator-research train. Логи лежат рядом с каталогом прогона, а не в нём: train.py
# отказывается переиспользовать каталог. artifacts/runs/<experiment>/<run_id>.{train,agl-server,agl-controller}.log
# Ключ AGL: AGL_KEY из окружения или случайный на прогон. Экспортируется, не печатается; agl-server и
# agl-controller получают ссылку ${oc.env:AGL_KEY} (Hydra берёт значение из окружения), поэтому ключа нет
# ни в argv, ни в overrides.yaml Hydra.
# Env: AGL_PORT (по умолчанию 18080), AGL_KEY, переменные ключей из конфига, RAY_local_fs_capacity_threshold,
#      PYTORCH_CUDA_ALLOC_CONF.
# Usage (repo root): bash scripts/train.sh CONFIG [--smoke] [--run-id ID]
# xtrace выключен даже при `bash -x`: ключи из env не должны попасть в вывод.
set +x
set -euo pipefail
source "$(dirname "$0")/lib.sh"

usage() { echo "usage: bash scripts/train.sh CONFIG [--smoke] [--run-id ID]" >&2; exit 2; }
CONFIG="${1:-}"
[ -n "$CONFIG" ] || usage
shift
SMOKE=""
RUN_ID=""
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE="--smoke" ;;
        --run-id) [ $# -ge 2 ] || usage; RUN_ID="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

EXP_NAME="$(cfg_get "$CONFIG" experiment.name)"
MODEL_PATH="$(cfg_get "$CONFIG" orchestrator.model_path)"
AGENT_CONCURRENCY="$(cfg_get "$CONFIG" training.agent_concurrency)"
require_env "$(cfg_get "$CONFIG" subagents.api_key_env)" "$(cfg_get "$CONFIG" judge.api_key_env)"

# doctor выходит с 1 при любом FAIL — set -e останавливает скрипт до запуска agl и Ray.
.venv/bin/orchestrator-research doctor --config "$CONFIG"

export AGL_KEY="${AGL_KEY:-$(openssl rand -hex 16)}"
AGL_PORT="${AGL_PORT:-18080}"
export AGL_SERVER_URL="http://127.0.0.1:$AGL_PORT"
# Общий сервер: всегда свой Ray-кластер (к чужому не цепляемся) и свой temp вместо общего /tmp/ray.
# Лимиты CPU/RAM — в configs/base.toml extra_overrides.
export RAY_ADDRESS=local RAY_TMPDIR="$HOME/.ray_tmp"
# Общий /data на ext1 заполнен на ~97%; при дефолтном пороге 0.95 Ray перестаёт спиллить объекты и прогон встаёт.
export RAY_local_fs_capacity_threshold="${RAY_local_fs_capacity_threshold:-0.99}"
# Проверка CUDA через NVML: is_available() не должна зафиксировать список видимых карт до того, как Ray раздаст GPU.
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$RAY_TMPDIR"

RUN_ID="${RUN_ID:-${SMOKE:+smoke-}$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="artifacts/runs/$EXP_NAME"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/$RUN_ID.train.log"
AGL_SERVER_PID=""
AGL_CONTROLLER_PID=""
log() { echo "$*" | tee -a "$TRAIN_LOG"; }

cleanup() {
    [ -n "$AGL_CONTROLLER_PID" ] && kill "$AGL_CONTROLLER_PID" 2>/dev/null || true
    [ -n "$AGL_SERVER_PID" ] && kill "$AGL_SERVER_PID" 2>/dev/null || true
    .venv/bin/ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

log "run: $RUN_ID  experiment: $EXP_NAME  log: $TRAIN_LOG"

# Шлюз. default_proxy.model_name обязан совпадать с путём модели актора verl (доки agl API gateway).
# key — литерал '${oc.env:AGL_KEY}' в одинарных кавычках: значение подставит Hydra, в argv его нет.
.venv/bin/python -m orchestrator_research.training.agl_server port="$AGL_PORT" key='${oc.env:AGL_KEY}' default_proxy.model_name="$MODEL_PATH" \
    >"$LOG_DIR/$RUN_ID.agl-server.log" 2>&1 </dev/null &
AGL_SERVER_PID=$!
wait_http "$AGL_SERVER_URL/healthz" 120 "$AGL_SERVER_PID" "$LOG_DIR/$RUN_ID.agl-server.log"
log "agl-server ready at $AGL_SERVER_URL"

# Контроллер: каждый роллаут — локальный подпроцесс, одновременно не больше AGENT_CONCURRENCY.
.venv/bin/agl-controller runner_type=local agl_server.url="$AGL_SERVER_URL" agl_server.key='${oc.env:AGL_KEY}' \
    local_runner.maximum_size="$AGENT_CONCURRENCY" \
    >"$LOG_DIR/$RUN_ID.agl-controller.log" 2>&1 </dev/null &
AGL_CONTROLLER_PID=$!
log "agl-controller started (pid $AGL_CONTROLLER_PID, maximum_size=$AGENT_CONCURRENCY)"

# Тренер. train.py берёт AGL_SERVER_URL и AGL_KEY из окружения.
.venv/bin/orchestrator-research train --config "$CONFIG" --run-id "$RUN_ID" $SMOKE 2>&1 | tee -a "$TRAIN_LOG"
log "training finished: artifacts/runs/$EXP_NAME/$RUN_ID"
