#!/bin/bash
set -euo pipefail

ROOT=/home/kan/Research/SHuBERT
LIST_DIR="${LIST_DIR:-$ROOT/how2sign/manifest_files}"
OUTPUT="${OUTPUT:-/mnt/e/datasets/how2sign/shubert/shubert_hidden_feats}"
CHECKPOINT="${CHECKPOINT:-$ROOT/weights/checkpoint_836_400000.pt}"
REPORTS="${REPORTS:-$ROOT/reports}"
SPLITS="${SPLITS:-val test train}"

source "$ROOT/.venv-shubert/bin/activate"
export PYTHONPATH="$ROOT/fairseq:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$REPORTS"

for split in $SPLITS; do
    mkdir -p "$OUTPUT/$split"
    echo "[$(date)] $split: starting extraction"
    python "$ROOT/features/shubert_inference.py" \
        --csv_path "$LIST_DIR/shubert_input_$split.tsv" \
        --checkpoint_path "$CHECKPOINT" \
        --output_dir "$OUTPUT/$split" \
        --failure_report "$REPORTS/${split}_failures.jsonl"
    echo "[$(date)] $split: extraction complete, verifying"
    python "$ROOT/features/verify_shubert_outputs.py" \
        --manifest "$LIST_DIR/shubert_input_$split.tsv" \
        --output_dir "$OUTPUT/$split" \
        --split "$split"
    echo "[$(date)] $split: verification passed"
done

echo "[$(date)] === STAGE 6 DONE ==="
