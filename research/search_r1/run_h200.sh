#!/usr/bin/env bash
# Official examples/search_r1/run.sh on two GPUs of a shared node. Only environment variables and config overrides
# are added; each override and its reason is listed in research/README.md. Extra arguments go to the trainer.
#
# Usage (from the repo root, retrieval service already up):
#   SEARCH_R1_DATA_DIR=... TRAIN_GPUS=2,4 bash research/search_r1/run_h200.sh
set -euo pipefail

DATA_DIR="${SEARCH_R1_DATA_DIR:?set SEARCH_R1_DATA_DIR to the data directory}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:?set TRAIN_GPUS to two free GPU indices, e.g. 2,4}"
n_gpus=$(tr ',' '\n' <<<"$TRAIN_GPUS" | grep -c .)

export AGL_SERVER_PORT="${AGL_SERVER_PORT:-18181}"
export AGL_KEY="${AGL_KEY:-$(python -c 'import secrets; print(secrets.token_hex(16))')}"
export SEARCH_R1_RETRIEVAL_URL="http://127.0.0.1:${SEARCH_R1_RETRIEVAL_PORT:-18000}/retrieve"
# A fresh local Ray cluster under our home; never attach to another user's cluster.
export RAY_ADDRESS=local
export RAY_TMPDIR="${RAY_TMPDIR:-$HOME/.ray_tmp}"
mkdir -p "$RAY_TMPDIR"
# Llama-3.2 is gated: HF_TOKEN comes from a file the user wrote, never from argv.
if [[ -f "$HOME/.hf_token" ]]; then
    source "$HOME/.hf_token"
fi

exec bash examples/search_r1/run.sh \
    --train-file "$DATA_DIR/train.parquet" \
    --val-file "$DATA_DIR/val_512.parquet" \
    trainer.n_gpus_per_node="$n_gpus" \
    data.train_batch_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    trainer.total_training_steps=150 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    'trainer.logger=[console]' \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.3}" \
    "$@"
