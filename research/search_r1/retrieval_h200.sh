#!/usr/bin/env bash
# Official examples/search_r1/retrieval_launch.sh on one chosen GPU of a shared node, listening on localhost only.
#
# Usage (from the repo root): SEARCH_R1_DATA_DIR=... RETRIEVAL_GPU=3 bash research/search_r1/retrieval_h200.sh
set -euo pipefail

: "${SEARCH_R1_DATA_DIR:?set SEARCH_R1_DATA_DIR to the data directory}"
export CUDA_VISIBLE_DEVICES="${RETRIEVAL_GPU:?set RETRIEVAL_GPU to one free GPU index}"
export SEARCH_R1_RETRIEVAL_HOST=127.0.0.1
export SEARCH_R1_RETRIEVAL_PORT="${SEARCH_R1_RETRIEVAL_PORT:-18000}"
export SEARCH_R1_RETRIEVER_DEVICE=cuda

if ! command -v conda >/dev/null 2>&1 && [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi

exec bash examples/search_r1/retrieval_launch.sh
