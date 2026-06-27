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

declare -a JOB_MODEL_IDS
declare -a JOB_CKPT_PATHS
declare -a JOB_SEEDS
declare -a WORKER_PIDS

for model_idx in "${!MODEL_IDS[@]}"; do
  for seed in "${RUN_SEEDS[@]}"; do
    JOB_MODEL_IDS+=("${MODEL_IDS[$model_idx]}")
    JOB_CKPT_PATHS+=("${CKPT_PATHS[$model_idx]}")
    JOB_SEEDS+=("$seed")
  done
done

run_job() {
  local gpu="$1"
  local model_id="$2"
  local ckpt="$3"
  local seed="$4"
  local run_name="brevo_${model_id}_seed${seed}"
  local log="$LOG_DIR/${run_name}.log"

  if [ -f "$log" ] && grep -q "Done. Results appended" "$log"; then
    echo "Skipping completed run: $run_name"
    return
  fi

  echo "GPU $gpu -> $run_name"
  WANDB_RUN_GROUP="$model_id" CUDA_VISIBLE_DEVICES="$gpu" \
    bash run_brevo.sh "$model_id" "$ckpt" "$seed" "$PROJECT" "$run_name" \
    > "$log" 2>&1
}

worker() {
  local gpu="$1"
  local job_idx="$2"
  local stride="${#GPUS[@]}"

  while [ "$job_idx" -lt "${#JOB_MODEL_IDS[@]}" ]; do
    run_job \
      "$gpu" \
      "${JOB_MODEL_IDS[$job_idx]}" \
      "${JOB_CKPT_PATHS[$job_idx]}" \
      "${JOB_SEEDS[$job_idx]}"
    job_idx=$((job_idx + stride))
  done
}

cleanup() {
  local pids
  pids="$(jobs -pr || true)"
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
  fi
}
trap cleanup INT TERM

echo "Queueing ${#JOB_MODEL_IDS[@]} BREVO runs across ${#GPUS[@]} GPUs (project: $PROJECT)..."

for slot in "${!GPUS[@]}"; do
  if [ "$slot" -lt "${#JOB_MODEL_IDS[@]}" ]; then
    worker "${GPUS[$slot]}" "$slot" &
    WORKER_PIDS+=("$!")
  fi
done

for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid"
done
echo "All BREVO model-folder queued runs complete."
