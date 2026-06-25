#!/bin/bash
# Run Brevo (topological sort) probe.
#
# Settings: very_hard preset, vocab/label space N=40, train graphs n~U[3,30],
# OOD eval graphs n in [31,40]. 10k training steps.
#
# Usage:
#   bash run_brevo.sh scratch                          # random init baseline
#   bash run_brevo.sh sort /path/to/ckpt.pth          # pretrained from sort task
#   bash run_brevo.sh sort /path/to/ckpt.pth <seed>   # with explicit seed

set -euo pipefail
cd "$(dirname "$0")"

# Activate venv (needed when script is launched as a background subprocess)
VENV="$(pwd)/../.venv"
# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

MODE="${1:-scratch}"
CKPT="${2:-}"
SEED="${3:-0}"
PROJECT="${4:-synthetic_playground}"

OUT_DIR="results/brevo"
mkdir -p "$OUT_DIR"

if [ -z "$CKPT" ] && [ "$MODE" != "scratch" ]; then
  echo "Usage: bash run_brevo.sh scratch"
  echo "       bash run_brevo.sh <tag> /path/to/checkpoint.pth"
  exit 1
fi

TAG="${MODE}_seed${SEED}"
EXTRA=""
if [ -n "$CKPT" ]; then
  EXTRA="--pretrained_path $CKPT --transfer attn,ffn,ln"
fi

WANDB_ENTITY=fjfehr python -m downstream.synthetic_playground.brevo.brevo_train \
  --difficulty   very_hard \
  --vocab_n      40 \
  --train_max_n  30 \
  --max_steps    10000 \
  --eval_steps   1000 \
  --bsz          128 \
  --lr           5e-4 \
  --seed         $SEED \
  $EXTRA \
  --results_csv  "$OUT_DIR/brevo_results.csv" \
  --report_to    wandb \
  --wandb_project "$PROJECT" \
  --wandb_name   "brevo_${TAG}" \
  --model_name   "$MODE"

echo "Done. Results appended to $OUT_DIR/brevo_results.csv"
