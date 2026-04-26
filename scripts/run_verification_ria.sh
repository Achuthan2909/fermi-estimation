#!/usr/bin/env bash
# Verification run for Ria — train (0-1408) + val (0-846), batches of 250
# Usage: bash scripts/run_verification_ria.sh
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

# --- TRAIN (1409 questions: offset 0-1408) ---
run_batch datasets/final/train.json    0  250  run/verification_train_ria_b1.json
run_batch datasets/final/train.json  250  250  run/verification_train_ria_b2.json
run_batch datasets/final/train.json  500  250  run/verification_train_ria_b3.json
run_batch datasets/final/train.json  750  250  run/verification_train_ria_b4.json
run_batch datasets/final/train.json 1000  250  run/verification_train_ria_b5.json
run_batch datasets/final/train.json 1250  159  run/verification_train_ria_b6.json

# --- VAL (847 questions: offset 0-846) ---
run_batch datasets/final/val.json    0  250  run/verification_val_ria_b1.json
run_batch datasets/final/val.json  250  250  run/verification_val_ria_b2.json
run_batch datasets/final/val.json  500  250  run/verification_val_ria_b3.json
run_batch datasets/final/val.json  750   97  run/verification_val_ria_b4.json

echo ""
echo "All Ria batches complete."
echo "Train files: run/verification_train_ria_b{1..6}.json"
echo "Val files:   run/verification_val_ria_b{1..4}.json"
