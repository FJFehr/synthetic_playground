#!/bin/bash
# Launch all 8 evaluation runs in parallel, one per GPU.
#
# 2 tasks (brevo, depo) x 2 checkpoints (step2500, step10000) x 2 seeds (0, 1)
#
# Usage: bash run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

CKPT2500="../pytorch_model_1_step2500.pth"
CKPT10000="../pytorch_model_1_step10000.pth"

echo "Launching 8 runs across GPUs 0-7..."

# BREVO — step2500
CUDA_VISIBLE_DEVICES=0 bash run_brevo.sh sort "$CKPT2500"  0 > logs/brevo_step2500_seed0.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 bash run_brevo.sh sort "$CKPT2500"  1 > logs/brevo_step2500_seed1.log  2>&1 &

# BREVO — step10000
CUDA_VISIBLE_DEVICES=2 bash run_brevo.sh sort "$CKPT10000" 0 > logs/brevo_step10000_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash run_brevo.sh sort "$CKPT10000" 1 > logs/brevo_step10000_seed1.log 2>&1 &

# DEPO — step2500
CUDA_VISIBLE_DEVICES=4 bash run_depo.sh stack "$CKPT2500"  0 > logs/depo_step2500_seed0.log   2>&1 &
CUDA_VISIBLE_DEVICES=5 bash run_depo.sh stack "$CKPT2500"  1 > logs/depo_step2500_seed1.log   2>&1 &

# DEPO — step10000
CUDA_VISIBLE_DEVICES=6 bash run_depo.sh stack "$CKPT10000" 0 > logs/depo_step10000_seed0.log  2>&1 &
CUDA_VISIBLE_DEVICES=7 bash run_depo.sh stack "$CKPT10000" 1 > logs/depo_step10000_seed1.log  2>&1 &

echo "All 8 runs launched. Tail logs with:"
echo "  tail -f logs/*.log"
echo ""
wait
echo "All runs complete."
