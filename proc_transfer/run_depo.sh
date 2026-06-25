#!/bin/bash
# Run DEPO multi-hop dereference probe.
#
# Usage:
#   bash run_depo.sh scratch                           # random init baseline
#   bash run_depo.sh stack /path/to/ckpt.pth          # pretrained init
#   bash run_depo.sh stack /path/to/ckpt.pth <seed>   # with explicit seed

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

# ---------- edit these paths ----------
SORT_CKPT=""   # e.g. pretrained_models/procedural/4_4_512/10ksteps.../sort/seed0/.../pytorch_model.pth
# --------------------------------------

OUT_DIR="results/depo"
mkdir -p "$OUT_DIR"

if [ -z "$CKPT" ] && [ "$MODE" != "scratch" ]; then
  echo "Usage: bash run_depo.sh scratch"
  echo "       bash run_depo.sh <task_name> /path/to/checkpoint.pth"
  exit 1
fi

TAG="${MODE}_seed${SEED}"
EXTRA=""
if [ -n "$CKPT" ]; then
  EXTRA="--pretrained_path $CKPT --transfer attn,ffn,ln"
fi

WANDB_ENTITY=fjfehr python plotting/depo_depth_test.py \
  --tag         "$TAG" \
  --max_hops    4 \
  --num_entities 10 \
  --ood_hops    0 \
  --max_steps   100000 \
  --eval_steps  2000 \
  --n_eval      500 \
  --bsz         128 \
  --lr          5e-4 \
  --seed        $SEED \
  $EXTRA \
  --report_to   wandb \
  --wandb_project "$PROJECT" \
  --model_name  "$MODE" \
  --out_json    "$OUT_DIR/depo_${TAG}.json"

echo "Done. Results in $OUT_DIR/depo_${TAG}.json"
