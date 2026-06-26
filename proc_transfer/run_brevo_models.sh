#!/bin/bash
# Queue the new models-folder BREVO experiment across 8 GPUs.
#
# 5 pretrained checkpoints x 3 finetuning seeds = 15 BREVO runs.
# The wandb model identifier is the checkpoint name stem, e.g. set_seed0.
# The queue keeps at most one run per GPU active at a time.
#
# Usage:
#   bash run_brevo_models.sh
#   bash run_brevo_models.sh <wandb_project>

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="${1:-synthetic_playground}"
LOG_DIR="logs/brevo_models"
mkdir -p "$LOG_DIR"

GPUS=(0 1 2 3 4 5 6 7)
RUN_SEEDS=(0 1 2)

MODEL_IDS=(set_seed0 set_seed1 set_seed42 sortseed0 sort_seed42)
CKPT_PATHS=(
  "../models/set_seed0_step10000.pth"
  "../models/set_seed1_step10000.pth"
  "../models/set_seed42_step10000.pth"
  "../models/sortseed0_step10000.pth"
  "../models/sort_seed42.pth"
)

declare -a SLOT_PIDS
declare -a SLOT_LOGS
declare -a SLOT_NAMES

cleanup() {
  local pids
  pids="$(jobs -pr || true)"
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
  fi
}
trap cleanup INT TERM

wait_for_slot() {
  local slot pid
  while true; do
    for slot in "${!GPUS[@]}"; do
      pid="${SLOT_PIDS[$slot]:-}"
      if [ -z "$pid" ]; then
        echo "$slot"
        return
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        if ! wait "$pid"; then
          echo "Run failed: ${SLOT_NAMES[$slot]} (log: ${SLOT_LOGS[$slot]})" >&2
          exit 1
        fi
        unset "SLOT_PIDS[$slot]"
        echo "$slot"
        return
      fi
    done
    sleep 10
  done
}

launch_run() {
  local slot="$1"
  local model_id="$2"
  local ckpt="$3"
  local seed="$4"
  local gpu="${GPUS[$slot]}"
  local run_name="brevo_${model_id}_seed${seed}"
  local log="$LOG_DIR/${run_name}.log"

  echo "GPU $gpu -> $run_name"
  WANDB_RUN_GROUP="$model_id" CUDA_VISIBLE_DEVICES="$gpu" \
    bash run_brevo.sh "$model_id" "$ckpt" "$seed" "$PROJECT" "$run_name" \
    > "$log" 2>&1 &

  SLOT_PIDS[$slot]=$!
  SLOT_LOGS[$slot]="$log"
  SLOT_NAMES[$slot]="$run_name"
}

echo "Queueing 15 BREVO runs across ${#GPUS[@]} GPUs (project: $PROJECT)..."

job_count=0
for model_idx in "${!MODEL_IDS[@]}"; do
  for seed in "${RUN_SEEDS[@]}"; do
    slot="$(wait_for_slot)"
    launch_run \
      "$slot" \
      "${MODEL_IDS[$model_idx]}" \
      "${CKPT_PATHS[$model_idx]}" \
      "$seed"
    job_count=$((job_count + 1))
  done
done

echo "Launched $job_count runs. Waiting for remaining jobs..."
for slot in "${!GPUS[@]}"; do
  pid="${SLOT_PIDS[$slot]:-}"
  if [ -n "$pid" ]; then
    if ! wait "$pid"; then
      echo "Run failed: ${SLOT_NAMES[$slot]} (log: ${SLOT_LOGS[$slot]})" >&2
      exit 1
    fi
  fi
done

echo "All BREVO model-folder queued runs complete."
