#!/bin/bash
# Launch the new models-folder BREVO experiment, one run per GPU.
#
# 5 pretrained checkpoints:
#   set:  seeds 0, 1, 42
#   sort: seeds 0, 42
#
# Usage:
#   bash run_brevo_models.sh
#   bash run_brevo_models.sh <wandb_project>

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="${1:-synthetic_playground}"
LOG_DIR="logs/brevo_models"
mkdir -p "$LOG_DIR"

SET_SEED0="../models/set_seed0_step10000.pth"
SET_SEED1="../models/set_seed1_step10000.pth"
SET_SEED42="../models/set_seed42_step10000.pth"
SORT_SEED0="../models/sortseed0_step10000.pth"
SORT_SEED42="../models/sort_seed42.pth"

echo "Launching 5 BREVO model-folder runs across GPUs 0-4 (project: $PROJECT)..."

WANDB_RUN_GROUP=set  CUDA_VISIBLE_DEVICES=0 bash run_brevo.sh set  "$SET_SEED0"  0  "$PROJECT" > "$LOG_DIR/brevo_set_seed0.log"  2>&1 &
WANDB_RUN_GROUP=set  CUDA_VISIBLE_DEVICES=1 bash run_brevo.sh set  "$SET_SEED1"  1  "$PROJECT" > "$LOG_DIR/brevo_set_seed1.log"  2>&1 &
WANDB_RUN_GROUP=set  CUDA_VISIBLE_DEVICES=2 bash run_brevo.sh set  "$SET_SEED42" 42 "$PROJECT" > "$LOG_DIR/brevo_set_seed42.log" 2>&1 &

WANDB_RUN_GROUP=sort CUDA_VISIBLE_DEVICES=3 bash run_brevo.sh sort "$SORT_SEED0"  0  "$PROJECT" > "$LOG_DIR/brevo_sort_seed0.log"  2>&1 &
WANDB_RUN_GROUP=sort CUDA_VISIBLE_DEVICES=4 bash run_brevo.sh sort "$SORT_SEED42" 42 "$PROJECT" > "$LOG_DIR/brevo_sort_seed42.log" 2>&1 &

echo "All 5 BREVO runs launched. Tail logs with:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
wait
echo "All BREVO model-folder runs complete."
