#!/usr/bin/env bash
# Train the Glimmer-attention + Qwen3-MoE (~1.9B total, ~180M active) model
# on the local ClimbMix parquet subset.
#
# Usage:
#   ./scripts/train_glimmer_qwen3_moe.sh
#   TRAIN_STEPS=2 LOG_FREQ=1 ./scripts/train_glimmer_qwen3_moe.sh
#   LOG_FILE=/tmp/run.log ./scripts/train_glimmer_qwen3_moe.sh
#   NGPU=4 EP_DEGREE=4 ATTN_BACKEND=flex ./scripts/train_glimmer_qwen3_moe.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

NGPU="${NGPU:-4}"
# Expert parallel all-to-all dominates this small model on PCIe. DP/FSDP only.
EP_DEGREE="${EP_DEGREE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
LOG_FREQ="${LOG_FREQ:-10}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-64}"
# FP8=0 keeps BF16. FP8=1 quantizes dense linears and MoE grouped GEMMs.
FP8="${FP8:-1}"
# Packed sequences make flex attention quadratic in the pack length. Varlen
# attends inside each document and is the fast path on SM120.
ATTN_BACKEND="${ATTN_BACKEND:-varlen}"
CLIMBMIX_DIR="${CLIMBMIX_DIR:-/home/sky/data/climbmix-100b}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT}/assets/hf/gpt2}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/glimmer_qwen3_moe}"
# stdout/stderr are appended here and still printed on the terminal.
# Set LOG_FILE= to disable the extra file and keep only the terminal.
LOG_FILE="${LOG_FILE-${OUTPUT_DIR}/train.log}"
PYTHON="${PYTHON:-/home/sky/llm-venv/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sky/llm-venv/bin/torchrun}"

export EP_DEGREE TRAIN_STEPS LOG_FREQ LOCAL_BATCH_SIZE FP8 ATTN_BACKEND CLIMBMIX_DIR TOKENIZER_DIR
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export LOG_RANK="${LOG_RANK:-0}"

if [[ -n "${LOG_FILE}" ]]; then
    mkdir -p "$(dirname "${LOG_FILE}")"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "Logging to ${LOG_FILE}"
fi

if [[ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
    echo "Downloading GPT-2 tokenizer into ${TOKENIZER_DIR}"
    mkdir -p "${TOKENIZER_DIR}"
    base="https://huggingface.co/gpt2/resolve/main"
    curl -fsSL "${base}/tokenizer.json" -o "${TOKENIZER_DIR}/tokenizer.json"
    curl -fsSL "${base}/tokenizer_config.json" -o "${TOKENIZER_DIR}/tokenizer_config.json"
    curl -fsSL "${base}/vocab.json" -o "${TOKENIZER_DIR}/vocab.json"
    curl -fsSL "${base}/merges.txt" -o "${TOKENIZER_DIR}/merges.txt"
fi

if [[ ! -f "${CLIMBMIX_DIR}/shard_00000.parquet" ]]; then
    echo "ClimbMix shards not found in ${CLIMBMIX_DIR}" >&2
    exit 1
fi

"${TORCHRUN}" \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${LOG_RANK}" \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --module torchtitan_recipes.models.glimmer_qwen3_moe \
    --config glimmer_qwen3_moe \
    --output-dir "${OUTPUT_DIR}"
