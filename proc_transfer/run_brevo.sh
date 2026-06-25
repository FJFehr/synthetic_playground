#!/bin/bash
# Run Brevo (topological sort) probe.
#
# Settings: very_hard preset, vocab/label space N=40, train graphs n~U[3,30],
# OOD eval graphs n in [31,40]. 10k training steps.
#
# Usage:
#   bash run_brevo.sh scratch                   # random init baseline
#   bash run_brevo.sh sort /path/to/ckpt.pth    # pretrained from sort task

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-scratch}"
CKPT="${2:-}"

OUT_DIR="results/brevo"
mkdir -p "$OUT_DIR"

if [ -z "$CKPT" ] && [ "$MODE" != "scratch" ]; then
  echo "Usage: bash run_brevo.sh scratch"
  echo "       bash run_brevo.sh <tag> /path/to/checkpoint.pth"
  exit 1
fi

TAG="${MODE}"
EXTRA=""
if [ -n "$CKPT" ]; then
  EXTRA="--pretrained_path $CKPT --transfer attn,ffn,ln"
fi

python -m downstream.synthetic_playground.brevo.brevo_train \
  --difficulty   very_hard \
  --vocab_n      40 \
  --train_max_n  30 \
  --max_steps    10000 \
  --eval_steps   1000 \
  --bsz          128 \
  --lr           5e-4 \
  --seed         0 \
  $EXTRA \
  --results_csv  "$OUT_DIR/brevo_results.csv" \
  --wandb_name   "brevo_${TAG}"

echo "Done. Results appended to $OUT_DIR/brevo_results.csv"
