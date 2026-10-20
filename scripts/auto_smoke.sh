#!/usr/bin/env bash
# Смоук без присмотра на ОБЩЕМ сервере: ждёт сборку flash-attn (tmux-сессия fa), ждёт карту со свободной
# памятью и свободную RAM (или выходит по таймауту), поднимает наши сервисы, doctor + smoke_train.sh.
# Сервисы останавливаются ВСЕГДА (trap EXIT), чтобы не держать память соседей.
# Env: NEED_GB, NEED_GPUS, NEED_RAM_GB, WAIT_HOURS.
# Usage (in tmux, with scripts/ram_guard.sh running): bash scripts/auto_smoke.sh experiments/000_smoke_local/config.toml
set -uo pipefail
source "$(dirname "$0")/lib.sh"
CONFIG="${1:?usage: auto_smoke.sh CONFIG}"
NEED_GB="${NEED_GB:-20}"     # свободных GB на КАЖДОЙ карте
NEED_GPUS="${NEED_GPUS:-1}"   # число карт (training.n_gpus)

while tmux has-session -t fa 2>/dev/null; do sleep 60; done
.venv/bin/python -c "import flash_attn" 2>/dev/null || { echo "flash-attn is not installed; see artifacts/flash_attn_build.log"; exit 1; }

NEED_RAM_GB="${NEED_RAM_GB:-30}"
WAIT_HOURS="${WAIT_HOURS:-12}"
deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
# Очередь: ждём, пока на общем сервере не освободятся NEED_GB на картах И NEED_RAM_GB RAM.
while true; do
    # NEED_GPUS самых свободных карт в режиме Default (Exclusive_Process не открыть); FREE — минимум по ним,
    # 0, если таких карт меньше NEED_GPUS.
    read -r GPU FREE < <(nvidia-smi --query-gpu=index,memory.total,memory.used,compute_mode --format=csv,noheader,nounits |
        awk -F', ' '$4=="Default" { printf "%.0f %s\n", ($2-$3)/1024, $1 }' | sort -rn | head -n "$NEED_GPUS" |
        awk -v need="$NEED_GPUS" '{ ids = ids (NR>1 ? "," : "") $2; if (NR==1 || $1<min) min=$1 }
             END { if (NR < need) min=0; printf "%s %d\n", ids, min }')
    RAM=$(free -g | awk '/Mem/{print $7}')
    [ "$FREE" -ge "$NEED_GB" ] && [ "$RAM" -ge "$NEED_RAM_GB" ] && break
    [ "$(date +%s)" -ge "$deadline" ] && { echo "gave up after ${WAIT_HOURS} h: best GPU $GPU ${FREE} GB, RAM ${RAM} GB"; exit 1; }
    echo "$(date -u +%T) waiting: best GPU $GPU ${FREE}/${NEED_GB} GB free, RAM ${RAM}/${NEED_RAM_GB} GB"
    sleep 120
done
export CUDA_VISIBLE_DEVICES="$GPU"
echo "using GPU $GPU ($FREE GB free on the weakest card)"

trap 'bash scripts/services.sh stop "$CONFIG"' EXIT
bash scripts/services.sh start "$CONFIG" || exit 1
.venv/bin/orchestrator-research doctor --config "$CONFIG" || exit 1
bash scripts/smoke_train.sh "$CONFIG"
