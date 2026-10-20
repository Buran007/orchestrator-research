#!/usr/bin/env bash
# Сторож RAM на ОБЩЕЙ машине. Каждые INTERVAL с: если доступной RAM меньше MIN_GB, гасит только НАШЕ
# (переданные tmux-сессии: C-c, через 20 с kill; затем наши nvcc/ray/vllm/agl по pkill -u $USER) и выходит,
# чтобы OOM killer не пришёл к соседям.
# Env: MIN_GB (по умолчанию 6), INTERVAL (по умолчанию 5).
# Usage (in tmux): MIN_GB=12 bash scripts/ram_guard.sh fa smoke auto
set -uo pipefail
MIN_GB="${MIN_GB:-6}"
INTERVAL="${INTERVAL:-5}"
SESSIONS=("$@")
LOG="$(dirname "$0")/../artifacts/ram_guard.log"
mkdir -p "$(dirname "$LOG")"
echo "$(date -u +%FT%TZ) guard started: min ${MIN_GB} GB, sessions: ${SESSIONS[*]}" >> "$LOG"
while true; do
    avail=$(free -g | awk '/Mem/{print $7}')
    if [ "$avail" -lt "$MIN_GB" ]; then
        echo "$(date -u +%FT%TZ) LOW RAM ${avail} GB: stopping our jobs" >> "$LOG"
        for s in "${SESSIONS[@]}"; do tmux send-keys -t "$s" C-c 2>/dev/null; done
        sleep 20
        for s in "${SESSIONS[@]}"; do tmux kill-session -t "$s" 2>/dev/null; done
        pkill -u "$USER" -f "nvcc|cicc|ptxas|ninja|raylet|gcs_server|vllm|agl-server|agl-controller" 2>/dev/null
        echo "$(date -u +%FT%TZ) stopped; available now $(free -g | awk '/Mem/{print $7}') GB" >> "$LOG"
        exit 0
    fi
    sleep "$INTERVAL"
done
