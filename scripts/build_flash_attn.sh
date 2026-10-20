#!/usr/bin/env bash
# Сборка flash-attn в .venv на ОБЩЕМ сервере с ограничением RAM: MAX_JOBS параллельных nvcc (~8 GB RAM каждый),
# ядра только под A100 (TORCH_CUDA_ARCH_LIST), низший приоритет CPU. Запускать в tmux вместе с ram_guard.sh.
# Env: MAX_JOBS (по умолчанию 2), TORCH_CUDA_ARCH_LIST (по умолчанию 8.0).
# Usage: MAX_JOBS=2 bash scripts/build_flash_attn.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"

PY=.venv/bin/python
CUDA_HOME="$("$PY" -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE

echo "building flash-attn 2.8.3: MAX_JOBS=$MAX_JOBS arch=$TORCH_CUDA_ARCH_LIST CUDA_HOME=$CUDA_HOME"
nice -n 19 uv pip install --python "$PY" "flash-attn==2.8.3" \
    --no-cache --no-binary flash-attn --no-build-isolation --no-deps
"$PY" -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
