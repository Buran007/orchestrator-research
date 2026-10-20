#!/usr/bin/env bash
# Только чтение: свободная память на каждой GPU и подсказка CUDA_VISIBLE_DEVICES с самыми свободными картами.
# Карты в режиме Exclusive_Process (заняты, например MPS на ext1) показываются, но не предлагаются: чужой процесс
# их не откроет ("CUDA-capable device(s) is/are busy or unavailable"), сколько бы памяти ни было свободно.
# Env: NEED_GPUS=N — предложить N самых свободных карт в режиме Default (для training.n_gpus = N).
# Usage: bash scripts/gpu_free.sh
set -euo pipefail

Q="$(nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu,compute_mode --format=csv,noheader,nounits)"
awk -F', ' '{ printf "GPU %s: free %5.1f GB of %5.1f GB, util %s%%, %s\n", $1, ($2-$3)/1024, $2/1024, $4, $5 }' <<<"$Q"
IDS="$(awk -F', ' '$5 == "Default" { printf "%d %s\n", $2-$3, $1 }' <<<"$Q" | sort -rn | head -n "${NEED_GPUS:-1}" |
  awk '{print $2}' | paste -sd, -)"
echo
echo "freest: export CUDA_VISIBLE_DEVICES=$IDS"
