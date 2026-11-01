#!/usr/bin/env bash
# Short single-GPU smoke run of one ARCH: 30 steps, 8-sequence microbatches,
# no validation or checkpoints. Usage: smoke_moe.sh <ARCH> [KEY=VALUE ...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
arch=$1; shift
out=/tmp/smoke_${arch}
rm -rf $out; mkdir -p $out
env "$@" PYTHONPATH="${ROOT}" PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  NGPU=1 LOG_FREQ=5 CLIMBMIX_DIR="${ROOT}/data/climbmix-100b" FLAVOR=d200m PLE=0 ARCH=$arch \
  HEAD_DIM=128 SWA_ROTARY_DIM=32 LR=1.4e-3 LOCAL_BATCH_SIZE=8 GLOBAL_BATCH_TOKENS=16384 TRAIN_STEPS=30 \
  "${HOME}/titan-venv/bin/torchrun" --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  -m torchtitan.train --module torchtitan_recipes.models.glimmer_dense --config glimmer_dense \
  --output-dir $out > $out/train.log 2>&1
grep -E "parameter count|tokens per train step|step: +(1|10|20|30)/" $out/train.log | cut -c1-230
