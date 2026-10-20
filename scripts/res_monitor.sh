#!/usr/bin/env bash
# Журнал ресурсов общего сервера, только чтение. Каждые INTERVAL с (по умолчанию 10) — строка на GPU в CSV:
#   ts, gpu, util_pct, total_mib, ours_mib, others_mib (= total - ours), host_used_gib, host_avail_gib, ours_rss_gib
# others во времени — нагрузка соседей без нашего кода; ours — что реально держат наши процессы.
# Больше ничего не трогает. Usage (in tmux): bash scripts/res_monitor.sh [artifacts/res_timeline.csv]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-artifacts/res_timeline.csv}"
INTERVAL="${INTERVAL:-10}"
mkdir -p "$(dirname "$OUT")"
[ -s "$OUT" ] || echo "ts,gpu,util_pct,total_mib,ours_mib,others_mib,host_used_gib,host_avail_gib,ours_rss_gib" > "$OUT"
while true; do
    ts="$(date -u +%FT%TZ)"
    mine="$(pgrep -u "$USER" -d'|' . || true)"
    read -r used avail < <(free -g | awk '/Mem/{print $3, $7}')
    rss="$(ps -u "$USER" -o rss= | awk '{s+=$1} END {printf "%.1f", s/1048576}')"
    # наша память по uuid карты: процессы nvidia-smi, чей pid принадлежит $USER
    declare -A ours=()
    while IFS=', ' read -r pid uuid mem; do
        [[ -n "$pid" && "$pid" =~ ^($mine)$ ]] && ours[$uuid]=$(( ${ours[$uuid]:-0} + mem ))
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null)
    while IFS=', ' read -r idx uuid util total; do
        o=${ours[$uuid]:-0}
        echo "$ts,$idx,$util,$total,$o,$(( total - o )),$used,$avail,$rss" >> "$OUT"
    done < <(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits)
    unset ours
    sleep "$INTERVAL"
done
