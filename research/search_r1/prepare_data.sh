#!/usr/bin/env bash
# Search-R1 data and retriever env for a shared node with little free disk.
# Same URLs, conda env and packages as examples/search_r1/data_process.sh. One difference: the official script keeps
# part_aa and part_ab and then concatenates them (~130 GB at peak); here part_ab is appended to part_aa as it
# downloads, so the peak is the final index size (~65 GB). Refuses to start if fewer than MIN_FREE_GB would remain.
#
# Usage: SEARCH_R1_DATA_DIR=/path/to/data bash research/search_r1/prepare_data.sh
set -euo pipefail

DATA_DIR="${SEARCH_R1_DATA_DIR:?set SEARCH_R1_DATA_DIR to the data directory}"
ENV_NAME="${SEARCH_R1_RETRIEVER_ENV:-retriever}"
PYTHON_VERSION="${SEARCH_R1_RETRIEVER_PYTHON:-3.10}"
PYTORCH_CUDA="${SEARCH_R1_PYTORCH_CUDA:-12.1}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"

WIKI_URL="https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz"
INDEX_A_URL="https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_aa"
INDEX_B_URL="https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_ab"
TRAIN_URL="https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main/train.parquet"
TEST_URL="https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main/test.parquet"

remote_size() {
    curl -sIL --fail "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n+0}'
}

mkdir -p "$DATA_DIR"
INDEX="$DATA_DIR/e5_Flat.index"
CORPUS="$DATA_DIR/wiki-18.jsonl"

# Disk check: what is still to download, plus the unpacked corpus (about 3x the archive).
need=0
if [[ ! -s "$INDEX" ]]; then
    need=$(( need + $(remote_size "$INDEX_A_URL") + $(remote_size "$INDEX_B_URL") ))
fi
if [[ ! -s "$CORPUS" ]]; then
    need=$(( need + 4 * $(remote_size "$WIKI_URL") ))
fi
free=$(df -B1 --output=avail "$DATA_DIR" | tail -1 | tr -dc 0-9)
need_gb=$(( need / 1073741824 )); free_gb=$(( free / 1073741824 ))
echo "need about ${need_gb} GB, free ${free_gb} GB, keep at least ${MIN_FREE_GB} GB free"
if (( free_gb - need_gb < MIN_FREE_GB )); then
    echo "not enough disk: would leave $(( free_gb - need_gb )) GB" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1 && [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required for the retriever env (install Miniforge into ~/miniforge3)" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"
if [[ "${SEARCH_R1_SKIP_RETRIEVER_INSTALL:-0}" != "1" ]]; then
    conda install -y \
        "pytorch==2.4.0" \
        "torchvision==0.19.0" \
        "torchaudio==2.4.0" \
        "pytorch-cuda=$PYTORCH_CUDA" \
        -c pytorch -c nvidia
    conda install -y -c pytorch -c nvidia "faiss-gpu=1.8.0"
    python -m pip install -U "transformers==4.57.1" datasets pyserini uvicorn fastapi tqdm
fi
conda deactivate

download() {  # url output; skips a finished file
    if [[ -s "$2" ]]; then echo "skip existing $2"; return; fi
    curl -L --fail --retry 5 -C - -o "$2.partial" "$1"
    mv "$2.partial" "$2"
}

append() {  # url file base_size; restarts the append from base_size on failure
    for attempt in 1 2 3 4 5; do
        truncate -s "$3" "$2"
        if curl -L --fail "$1" >> "$2"; then return; fi
        echo "append failed (attempt $attempt), retrying" >&2
        sleep 10
    done
    echo "could not download $1" >&2
    exit 1
}

if [[ ! -s "$INDEX" ]]; then
    size_a=$(remote_size "$INDEX_A_URL")
    download "$INDEX_A_URL" "$INDEX.part_aa"
    [[ $(stat -c %s "$INDEX.part_aa") -eq $size_a ]] || { echo "part_aa size mismatch" >&2; exit 1; }
    append "$INDEX_B_URL" "$INDEX.part_aa" "$size_a"
    mv "$INDEX.part_aa" "$INDEX"
fi

if [[ ! -s "$CORPUS" ]]; then
    download "$WIKI_URL" "$CORPUS.gz"
    gzip -dc "$CORPUS.gz" > "$CORPUS.partial"
    mv "$CORPUS.partial" "$CORPUS"
    rm -f "$CORPUS.gz"
fi

download "$TRAIN_URL" "$DATA_DIR/train.parquet"
download "$TEST_URL" "$DATA_DIR/test.parquet"

echo "Search-R1 data ready in $DATA_DIR"
df -h "$DATA_DIR" | tail -1
