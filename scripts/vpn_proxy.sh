#!/usr/bin/env bash
# Исходящий прокси в user space для общего сервера (ADR 0007): xray-core в tmux-сессии "xray", HTTP-прокси
# только на 127.0.0.1:$PROXY_PORT. Без root, TUN и смены маршрутов: через него идут лишь процессы с HTTPS_PROXY
# (scripts/proxy_env.sh, включается USE_PROXY=1 в lib.sh); остальных на машине это не касается.
# Подписку читает scripts/xray_config.py из ~/.vpn_sub или ~/.vpn_sub_body; она никогда не печатается.
#   start   один раз скачать xray (с проверкой sha256), затем перебирать серверы подписки, пока $CHECK_URL
#           не ответит 200 через прокси; оставить первый рабочий
#   stop    убить нашу tmux-сессию "xray"
#   status  сессия, HTTP-код $CHECK_URL с прокси и без, страна выхода через прокси
# Env: XRAY_VERSION, PROXY_PORT (по умолчанию 18090), CHECK_URL.
# Usage (repo root): bash scripts/vpn_proxy.sh {start|stop|status}
set -euo pipefail
source "$(dirname "$0")/lib.sh"
# Сам скрипт ходит в интернет напрямую (к прокси — только явным -x), даже при USE_PROXY=1.
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy

XRAY_VERSION="${XRAY_VERSION:-v26.3.27}"
PROXY_PORT="${PROXY_PORT:-18090}"
CHECK_URL="${CHECK_URL:-https://openrouter.ai/api/v1/models}"
XRAY_DIR="$HOME/.xray"
XRAY_BIN="$XRAY_DIR/xray"
PROXY="http://127.0.0.1:$PROXY_PORT"

code_via() { curl -s -o /dev/null -w "%{http_code}" -m 15 "$@" "$CHECK_URL" || true; }

install_xray() {
    [ -x "$XRAY_BIN" ] && return 0
    mkdir -p -m 700 "$XRAY_DIR"
    local base="https://github.com/XTLS/Xray-core/releases/download/$XRAY_VERSION/Xray-linux-64.zip"
    curl -sSfL -o "$XRAY_DIR/xray.zip" "$base"
    curl -sSfL -o "$XRAY_DIR/xray.zip.dgst" "$base.dgst"
    local want got
    want="$(grep -i '^SHA2-256=' "$XRAY_DIR/xray.zip.dgst" | awk '{print $2}')"
    got="$(sha256sum "$XRAY_DIR/xray.zip" | awk '{print $1}')"
    [ -n "$want" ] && [ "$want" = "$got" ] || { echo "error: xray.zip sha256 mismatch" >&2; exit 1; }
    python3 -m zipfile -e "$XRAY_DIR/xray.zip" "$XRAY_DIR"
    chmod 700 "$XRAY_BIN"
    rm -f "$XRAY_DIR/xray.zip" "$XRAY_DIR/xray.zip.dgst"
    echo "xray $XRAY_VERSION installed in $XRAY_DIR (sha256 ok)"
}

start() {
    if tmux has-session -t xray 2>/dev/null; then
        echo "xray already running; status:"; status; return 0
    fi
    if ss -ltn | awk '{print $4}' | grep -qE "[:.]$PROXY_PORT\$"; then
        echo "error: port $PROXY_PORT is busy (someone else's); set PROXY_PORT" >&2; exit 1
    fi
    install_xray
    local listing i code
    listing="$(python3 scripts/xray_config.py list)"
    echo "$listing" >&2
    if ! awk -F'\t' '$2 != "stub"' <<<"$listing" | grep -q .; then
        echo "error: the provider returned only messages (see above), no real servers" >&2; exit 1
    fi
    for i in $(awk -F'\t' '$2 != "stub" && $2 != "unparsed" {print $1}' <<<"$listing"); do
        python3 scripts/xray_config.py write "$i" --port "$PROXY_PORT" >/dev/null
        "$XRAY_BIN" run -test -c "$XRAY_DIR/config.json" >/dev/null 2>&1 || { echo "server $i: config rejected"; continue; }
        tmux new -d -s xray "$XRAY_BIN run -c $XRAY_DIR/config.json > $XRAY_DIR/xray.log 2>&1"
        sleep 3
        code="$(code_via -x "$PROXY")"
        if [ "$code" = "200" ]; then
            echo "server $i: $CHECK_URL -> 200 through $PROXY; keeping it"
            return 0
        fi
        echo "server $i: $CHECK_URL -> $code; trying next"
        tmux kill-session -t xray 2>/dev/null || true
    done
    echo "error: no server from the subscription reached $CHECK_URL" >&2
    exit 1
}

stop() { tmux kill-session -t xray 2>/dev/null && echo "xray stopped" || echo "xray: not running"; }

status() {
    tmux has-session -t xray 2>/dev/null && echo "session xray: running" || echo "session xray: not running"
    echo "direct:      $CHECK_URL -> $(code_via)"
    echo "via proxy:   $CHECK_URL -> $(code_via -x "$PROXY")"
    echo "egress via proxy: $(curl -s -m 15 -x "$PROXY" https://ipinfo.io/country || echo '?')"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) echo "usage: bash scripts/vpn_proxy.sh {start|stop|status}" >&2; exit 2 ;;
esac
