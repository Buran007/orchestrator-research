#!/usr/bin/env bash
# Подготовка сервера для обучения (Linux, NVIDIA GPU, CUDA 12.9 или 13.0, установлен uv).
#
# Порядок:
#   1. .venv на Python 3.12 (uv);
#   2. GPU-стек в порядке, который Agent Lightning v1.0.2 требует для verl==0.8.0:
#      vllm, затем verl, затем flash-attn из исходников (как в agl scripts/setup_verl.sh);
#   3. сам проект: uv pip install -e ".[benchmark,train,dev]";
#   4. pytest и validate-config на конфиге эксперимента;
#   5. версии ключевых пакетов.
#
# Env (необязательные):
#   CUDA_VARIANT=cu130|cu129   вариант колёс (agl рекомендует cu130 для verl 0.8.0; по умолчанию cu130)
#   VLLM_VERSION=0.22.0        пин vLLM (как в agl v1.0.2 scripts/setup_verl.sh для verl 0.8.0 и в прошлом проекте;
#                              в install-доке agl указана 0.20.2 - см. docs/deployment.md)
#
# Usage (from the repo root): bash scripts/bootstrap_server.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"

CUDA_VARIANT="${CUDA_VARIANT:-cu130}"
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
VERL_VERSION="0.8.0"
FLASH_ATTN_VERSION="2.8.3"
CONFIG="experiments/template/config.toml"

if [ "$(uname -s)" != "Linux" ]; then
    echo "error: training runs on Linux only (agl-controller local runner needs POSIX process groups)" >&2
    exit 1
fi
case "$CUDA_VARIANT" in cu129|cu130) ;; *) echo "error: CUDA_VARIANT must be cu129 or cu130" >&2; exit 2 ;; esac
command -v uv >/dev/null || { echo "error: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

if [ ! -x "$PY" ]; then
    uv venv --python 3.12 .venv
fi
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PYVER" != "3.12" ]; then
    echo "error: .venv uses Python $PYVER; remove .venv and rerun (training needs 3.12)" >&2
    exit 1
fi

echo "== 1/4 vLLM $VLLM_VERSION ($CUDA_VARIANT wheels)"
uv pip install --python "$PY" \
    "vllm==$VLLM_VERSION" \
    --torch-backend="$CUDA_VARIANT" \
    --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/$CUDA_VARIANT" \
    --extra-index-url "https://download.pytorch.org/whl/$CUDA_VARIANT" \
    --index-strategy unsafe-best-match

echo "== 2/4 verl $VERL_VERSION"
uv pip install --python "$PY" "verl[mcore]==$VERL_VERSION"

if [ "$CUDA_VARIANT" = "cu130" ]; then
    # flash-attn собирается под CUDA-рантайм torch; тулчейн CUDA 13.0 из pip, чтобы не было рассинхрона
    # с системной CUDA 12.x (те же шаги, что в agl setup_verl.sh).
    uv pip install --python "$PY" \
        "nvidia-cuda-nvcc>=13.0,<13.1" "nvidia-cuda-crt>=13.0,<13.1" "nvidia-nvvm>=13.0,<13.1" \
        "nvidia-cuda-cccl>=13.0,<13.1" "nvidia-cuda-runtime>=13.0,<13.1"
    CUDA_HOME="$("$PY" -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
    ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0}"
fi

echo "== 3/4 flash-attn $FLASH_ATTN_VERSION (built from source; 10-30 min)"
FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --python "$PY" \
    "flash-attn==$FLASH_ATTN_VERSION" --force-reinstall --no-cache --no-binary flash-attn \
    --no-build-isolation --no-deps

echo "== 4/4 project: agentlightning==1.0.2 comes from the train extra"
uv pip install --python "$PY" -e ".[benchmark,train,dev]"

echo "== checks"
"$PY" -m pytest -q
.venv/bin/orchestrator-research validate-config --config "$CONFIG"

echo "== versions"
"$PY" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
for name in ("torch", "vllm", "verl", "agentlightning", "transformers", "flash-attn"):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")
PY
echo "bootstrap finished; next: export OPENROUTER_API_KEY, then bash scripts/services.sh start $CONFIG"
