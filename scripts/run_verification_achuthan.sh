#!/usr/bin/env bash
# Verification run for Achuthan — train (1409-2817) + val (847-1692), batches of 250
# Usage: bash scripts/run_verification_achuthan.sh
set -euo pipefail

source .env

if [ -z "${TINKER_API_KEY:-}" ]; then
  echo "ERROR: TINKER_API_KEY not set. Run: source .env"
  exit 1
fi

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_TOKENS=2048
MAX_SEARCH_QUERIES=3
CONCURRENCY=4
SCRIPT="src/verifier/run_verifier_cached.py"

run_batch() {
  local dataset=$1 offset=$2 limit=$3 output=$4
  echo ""
  echo "=========================================="
  echo "BATCH: $output  (offset=$offset limit=$limit)"
  echo "=========================================="
  python $SCRIPT \
    --dataset "$dataset" \
    --offset "$offset" \
    --limit "$limit" \
    --model-a "$MODEL" \
    --max-tokens $MAX_TOKENS \
    --max-search-queries $MAX_SEARCH_QUERIES \
    --concurrency $CONCURRENCY \
    --output "$output"
  echo "DONE: $output"
}

# --- TRAIN (1409 questions: offset 1409-2817) ---
run_batch datasets/final/train.json 1409  250  run/verification_train_achuthan_b1.json
run_batch datasets/final/train.json 1659  250  run/verification_train_achuthan_b2.json
run_batch datasets/final/train.json 1909  250  run/verification_train_achuthan_b3.json
run_batch datasets/final/train.json 2159  250  run/verification_train_achuthan_b4.json
run_batch datasets/final/train.json 2409  250  run/verification_train_achuthan_b5.json
run_batch datasets/final/train.json 2659  159  run/verification_train_achuthan_b6.json

# --- VAL (846 questions: offset 847-1692) ---
run_batch datasets/final/val.json  847  250  run/verification_val_achuthan_b1.json
run_batch datasets/final/val.json 1097  250  run/verification_val_achuthan_b2.json
run_batch datasets/final/val.json 1347  250  run/verification_val_achuthan_b3.json
run_batch datasets/final/val.json 1597   96  run/verification_val_achuthan_b4.json

echo ""
echo "All Achuthan batches complete."
echo "Train files: run/verification_train_achuthan_b{1..6}.json"
echo "Val files:   run/verification_val_achuthan_b{1..4}.json"
