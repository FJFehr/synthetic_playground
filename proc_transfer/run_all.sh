#!/bin/bash
# Launch all 8 evaluation runs in parallel, one per GPU.
#
# 2 tasks (brevo, depo) x 2 pretrained models (sort, stack) x 2 seeds (0, 1)
#
# Usage: bash run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

CKPT_SORT="../sort_step10000.pth"
CKPT_STACK="../stack_step10000.pth"
PROJECT="synthetic_playground"

echo "Launching 8 runs across GPUs 0-7 (project: $PROJECT)..."

# BREVO — sort model
WANDB_RUN_GROUP=sort  CUDA_VISIBLE_DEVICES=0 bash run_brevo.sh sort  "$CKPT_SORT"  0 "$PROJECT" > logs/brevo_sort_seed0.log  2>&1 &
WANDB_RUN_GROUP=sort  CUDA_VISIBLE_DEVICES=1 bash run_brevo.sh sort  "$CKPT_SORT"  1 "$PROJECT" > logs/brevo_sort_seed1.log  2>&1 &

# BREVO — stack model
WANDB_RUN_GROUP=stack CUDA_VISIBLE_DEVICES=2 bash run_brevo.sh stack "$CKPT_STACK" 0 "$PROJECT" > logs/brevo_stack_seed0.log 2>&1 &
WANDB_RUN_GROUP=stack CUDA_VISIBLE_DEVICES=3 bash run_brevo.sh stack "$CKPT_STACK" 1 "$PROJECT" > logs/brevo_stack_seed1.log 2>&1 &

# DEPO — sort model
WANDB_RUN_GROUP=sort  CUDA_VISIBLE_DEVICES=4 bash run_depo.sh  sort  "$CKPT_SORT"  0 "$PROJECT" > logs/depo_sort_seed0.log   2>&1 &
WANDB_RUN_GROUP=sort  CUDA_VISIBLE_DEVICES=5 bash run_depo.sh  sort  "$CKPT_SORT"  1 "$PROJECT" > logs/depo_sort_seed1.log   2>&1 &

# DEPO — stack model
WANDB_RUN_GROUP=stack CUDA_VISIBLE_DEVICES=6 bash run_depo.sh  stack "$CKPT_STACK" 0 "$PROJECT" > logs/depo_stack_seed0.log  2>&1 &
WANDB_RUN_GROUP=stack CUDA_VISIBLE_DEVICES=7 bash run_depo.sh  stack "$CKPT_STACK" 1 "$PROJECT" > logs/depo_stack_seed1.log  2>&1 &

echo "All 8 runs launched. Tail logs with:"
echo "  tail -f logs/*.log"
echo ""
wait
echo "All runs complete."
