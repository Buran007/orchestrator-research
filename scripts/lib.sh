#!/usr/bin/env bash
# Общие функции для scripts/*.sh. Только source, не запускать; при подключении делает cd в корень репозитория.
# Секреты не печатаются: require_env называет только имя переменной; ключ AGL доходит до agl-server и
# agl-controller ссылкой Hydra ${oc.env:AGL_KEY} (см. train.sh).
# Env: USE_PROXY, OUR_HF_DATASETS_CACHE.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# torch.compile в vLLM нужен `ninja` из venv; uv лежит в ~/.local/bin.
export PATH="$PWD/.venv/bin:$HOME/.local/bin:$PATH"
PY=".venv/bin/python"
# На ext1 общий HF-кэш в /data/hf_cache. Модели оттуда делить можно, а подготовленные датасеты залочены тем,
# кто их собрал (PermissionError на *_builder.lock), поэтому кэш датасетов — в нашем home.
export HF_DATASETS_CACHE="${OUR_HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
# USE_PROXY=1: запросы к API (OpenRouter) идут через наш xray-прокси в user space (scripts/vpn_proxy.sh, ADR 0007).
if [ "${USE_PROXY:-0}" = "1" ]; then
    source scripts/proxy_env.sh
fi

# wait_http URL TIMEOUT_S PID LOGFILE: ждать ответа URL; сразу ошибка, если PID умер (хвост лога в stderr).
wait_http() {
    local url="$1" timeout="$2" pid="$3" log="$4" i
    for ((i = 0; i < timeout; i += 2)); do
        curl -sf -m 5 "$url" >/dev/null 2>&1 && return 0
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "error: process $pid exited before $url answered; tail of $log:" >&2
            tail -n 20 "$log" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "error: $url not ready after ${timeout}s; tail of $log:" >&2
    tail -n 20 "$log" >&2 || true
    return 1
}

# stop_pid PIDFILE: SIGTERM, до 30 с на завершение, затем SIGKILL; pid-файл удаляется.
stop_pid() {
    local pid_file="$1" name pid
    name="$(basename "$pid_file" .pid)"
    if [ ! -f "$pid_file" ]; then
        echo "$name: no pid file"
        return 0
    fi
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        echo "$name: stopped (pid $pid)"
    else
        echo "$name: pid $pid is not running"
    fi
    rm -f "$pid_file"
}

# cfg_get CONFIG DOTTED.KEY: одно значение из провалидированного конфига (через load_config, не парсинг TOML).
cfg_get() {
    "$PY" -c 'import sys
from functools import reduce
from orchestrator_research.config import load_config
print(reduce(getattr, sys.argv[2].split("."), load_config(sys.argv[1])))' "$1" "$2"
}

# url_host URL / url_port URL: хост и порт http URL (порт должен быть указан явно).
url_host() { sed -E 's#^[a-z]+://([^:/]+).*#\1#' <<<"$1"; }
url_port() { sed -E 's#^[a-z]+://[^/]*:([0-9]+).*#\1#' <<<"$1"; }

# is_local_url URL: успех только для хоста 127.0.0.1 или localhost — такой сервер поднимаем мы сами.
is_local_url() {
    case "$(url_host "$1")" in
        127.0.0.1 | localhost) return 0 ;;
        *) return 1 ;;
    esac
}

# require_env NAME...: exit 1, если переменная не задана или пуста; пустые имена пропускаются. Значения не печатаются.
require_env() {
    local name missing=0
    for name in "$@"; do
        [ -n "$name" ] || continue
        if [ -z "${!name:-}" ]; then
            echo "error: export $name first (its value is not printed)" >&2
            missing=1
        fi
    done
    [ "$missing" -eq 0 ] || exit 1
}
