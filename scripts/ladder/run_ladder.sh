#!/usr/bin/env bash
# Run Glimmer dense / PLE / Engram ladder steps on one 4xH100 node, one at a time.
#
#   nohup ./scripts/ladder/run_ladder.sh scripts/ladder/plans/dense.txt \
#       >> ~/ladder.out 2>&1 < /dev/null &
#
# Each plan line is "<flavor> <variant> <micro_batch> [KEY=VALUE ...]", where
# variant is 0 (dense), ple or engram, micro_batch is LOCAL_BATCH_SIZE, and the
# optional KEY=VALUE pairs are extra environment for that run only. PLE and
# Engram are synchronous (PLE_ASYNC_TRAINING=0). The recipe fixes the global
# batch per flavor and validates and checkpoints by tokens, so every run's val
# points and snapshots land on the same token counts.
#
# A finished run leaves outputs/ladder/<name>/DONE and is skipped; an
# interrupted run resumes from its latest checkpoint when relaunched. A run
# that hits OOM is retried with half the microbatch (same tokens per step).
#
# With R2 credentials ($R2_ENV), logs and TensorBoard go to $R2_DEST/<name>/
# every 30 min, and snapshot checkpoints (doubling token counts from 2^28, plus
# the final step) to $R2_DEST/<name>/checkpoint/step-N, after which the local
# copy is deleted. Without credentials everything stays local.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAN="$(realpath "${1:?usage: run_ladder.sh <plan.txt>}")"
cd "${ROOT}"

TITAN_VENV="${TITAN_VENV:-${HOME}/titan-venv}"
R2_ENV="${R2_ENV:-${HOME}/r2-credentials.env}"
R2_DEST="${R2_DEST:-r2:physiofm-shards/lsq_dev/torchtitan-ladder}"
RCLONE="${RCLONE:-$(command -v rclone || echo "${HOME}/rclone-v1.75.2-linux-amd64/rclone")}"
STAGE="${HOME}/ladder_sync_stage"

export PYTHONPATH="${ROOT}" PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export NGPU="${NGPU:-4}" LOG_FREQ="${LOG_FREQ:-10}" PLE_ASYNC_TRAINING=0 CKPT_KEEP=2 LOG_RANK=0
export CLIMBMIX_DIR="${CLIMBMIX_DIR:-${ROOT}/data/climbmix-100b}"

have_r2() { [[ -f "${R2_ENV}" && -x "${RCLONE}" ]]; }

rclone_r2() {
  (
    source "${R2_ENV}"
    export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
      RCLONE_CONFIG_R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \
      RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
      RCLONE_CONFIG_R2_ENDPOINT=$R2_ENDPOINT_URL
    "${RCLONE}" "$@"
  )
}

sync_r2() {  # tensorboard, structured logs and stdout; checkpoints go via upload_snapshots
  have_r2 || return 0
  # Files are still being appended, so upload a frozen local copy.
  rm -rf "${STAGE}"; mkdir -p "${STAGE}"
  rsync -a --exclude checkpoint --exclude comm_traces --exclude .uploaded outputs/ladder/ "${STAGE}/"
  rclone_r2 copy "${STAGE}" "${R2_DEST}" -q || echo "[ladder] R2 log sync failed (non-fatal)"
}

# Snapshots are doubling token counts from 2^28 (the recipe's purge_exempt) and
# the final step once DONE. After upload a snapshot is deleted locally once a
# newer complete checkpoint exists, so resuming never depends on R2. Off-grid
# checkpoints older than the newest two are deleted too.
upload_snapshots() {  # out
  have_r2 || return 0
  local out=$1 tps step d newest second tokens snap
  tps=$(grep -oE "[0-9]+ tokens per train step" $out/train.log 2>/dev/null | tail -1 | grep -oE "^[0-9]+")
  [[ -z $tps ]] && return
  mkdir -p $out/.uploaded
  newest=$(ls -d $out/checkpoint/step-* 2>/dev/null | sed 's/.*step-//' | sort -n | tail -1)
  second=$(ls -d $out/checkpoint/step-* 2>/dev/null | sed 's/.*step-//' | sort -n | tail -2 | head -1)
  for d in $(ls -d $out/checkpoint/step-* 2>/dev/null); do
    step=${d##*step-}
    [[ -f $d/.metadata || -f $d/model.safetensors.index.json ]] || continue
    tokens=$((step * tps)); snap=0
    (( tokens >= 2**28 && (tokens & (tokens - 1)) == 0 )) && snap=1
    [[ -f $out/DONE && $step == $newest ]] && snap=1
    if [[ $snap == 0 ]]; then
      [[ $step != $newest && $step != $second ]] && rm -rf $d
      continue
    fi
    if [[ ! -f $out/.uploaded/step-$step ]]; then
      if rclone_r2 copy $d "${R2_DEST}/${out##*/}/checkpoint/step-$step" -q \
          && rclone_r2 check $d "${R2_DEST}/${out##*/}/checkpoint/step-$step" --one-way --size-only -q; then
        echo "$tokens" > $out/.uploaded/step-$step
        echo "[ladder] $(date -u +%FT%TZ) uploaded ${out##*/} step-$step ($tokens tokens)"
      else
        echo "[ladder] upload of ${out##*/} step-$step failed, will retry"; continue
      fi
    fi
    [[ $step != $newest ]] && rm -rf $d
  done
}

train() {  # flavor variant micro out extra_env...
  local flavor=$1 variant=$2 mb=$3 out=$4; shift 4
  env "$@" FLAVOR=$flavor PLE=$variant LOCAL_BATCH_SIZE=$mb \
    "${TITAN_VENV}/bin/torchrun" --nproc_per_node=$NGPU --rdzv_backend c10d \
      --rdzv_endpoint=localhost:0 --local-ranks-filter $LOG_RANK --role rank --tee 3 \
      -m torchtitan.train --module torchtitan_recipes.models.glimmer_dense \
      --config glimmer_dense --output-dir $out >> $out/train.log 2>&1 &
  local pid=$! ticks=0
  while kill -0 $pid 2>/dev/null; do
    sleep 60; ticks=$((ticks + 1))
    (( ticks % 5 == 0 )) && upload_snapshots $out
    (( ticks % 30 == 0 )) && sync_r2
  done
  wait $pid
}

# A training process left by an earlier runner keeps the GPUs; wait for it and
# keep uploading its snapshots instead of starting a second job.
waits=0
while pgrep -f "[t]orchtitan.train" > /dev/null; do
  sleep 30; waits=$((waits + 1))
  if (( waits % 10 == 0 )); then
    for out in outputs/ladder/*/; do upload_snapshots ${out%/}; done
  fi
  (( waits % 60 == 0 )) && sync_r2
done

while read -r flavor variant mb extra; do
  [[ -z "${flavor}" || "${flavor}" == \#* ]] && continue
  name=${flavor}_$([[ $variant == 0 ]] && echo dense || echo $variant)
  # A non-default PLE/Engram width (recipe default 256) gets its own run name.
  dim=$(grep -oE "PLE_DIM=[0-9]+" <<< "${extra}" | cut -d= -f2)
  [[ -n $dim && $dim != 256 ]] && name=${name}${dim}
  out=outputs/ladder/$name
  if [[ ! -f $out/DONE ]] && tail -n 50 $out/train.log 2>/dev/null | grep -q "Training completed"; then
    touch $out/DONE; echo "[ladder] $(date -u +%FT%TZ) done $name (adopted)"
    upload_snapshots $out; sync_r2
  fi
  if [[ -f $out/DONE ]]; then echo "[ladder] skip $name (done)"; continue; fi
  mkdir -p $out
  while :; do
    echo "[ladder] $(date -u +%FT%TZ) start $name micro=$mb ${extra}"
    echo "===== $(date -u +%FT%TZ) launch micro=$mb ${extra} =====" >> $out/train.log
    # shellcheck disable=SC2086
    train $flavor $variant $mb $out ${extra}; rc=$?
    if [[ $rc == 0 ]] && tail -n 50 $out/train.log | grep -q "Training completed"; then
      touch $out/DONE; echo "[ladder] $(date -u +%FT%TZ) done $name"; break
    elif [[ $mb -gt 8 ]] && tail -n 300 $out/train.log | grep -qiE "OutOfMemory|out of memory"; then
      mb=$((mb / 2)); echo "[ladder] $(date -u +%FT%TZ) OOM $name, retry micro=$mb"
    else
      echo "[ladder] $(date -u +%FT%TZ) FAILED $name rc=$rc, see $out/train.log"; break
    fi
  done
  upload_snapshots $out
  sync_r2
done < "${PLAN}"
echo "[ladder] $(date -u +%FT%TZ) all runs processed"
