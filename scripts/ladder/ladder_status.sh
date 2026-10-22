#!/usr/bin/env bash
# One-screen status of the ladder runs on this node.
#
#   ./scripts/ladder/ladder_status.sh [runner_log, default ~/ladder.out]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${1:-${HOME}/ladder.out}"
cd "${ROOT}/outputs/ladder" 2>/dev/null || { echo "no runs yet"; exit; }
echo "== queue (${LOG})"; grep "^\[ladder\]" "${LOG}" 2>/dev/null | tail -5
echo; echo "== runs"
for d in */; do
  r=${d%/}; s=$([[ -f $r/DONE ]] && echo DONE || echo running/pending)
  last=$(grep -E "step: *[0-9]+/" $r/train.log 2>/dev/null | tail -1 | sed "s/\x1b\[[0-9;]*m//g" \
    | grep -oE "step: *[0-9]+/[0-9]+ \([^)]*\)|loss: *[0-9.]+|tps: *[0-9,]+" | tr "\n" " ")
  val=$(grep "val step" $r/train.log 2>/dev/null | tail -1 | sed "s/\x1b\[[0-9;]*m//g" \
    | grep -oE "val step: *[0-9]+|val_loss: *[0-9.]+" | tr "\n" " ")
  snaps=$(ls $r/.uploaded 2>/dev/null | sort -t- -k2 -n | tr "\n" " ")
  printf "%-14s %-15s %s\n%14s last val: %s\n%14s R2 snapshots: %s\n" "$r" "$s" "$last" "" "$val" "" "$snaps"
done
echo; echo "== GPUs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
