#!/bin/bash
set -u

VENV=/home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate
SCRIPT=/home/kan/Research/SHuBERT/dataset/crop_hands.py
LIST_DIR=/home/kan/Research/SHuBERT/how2sign/manifest_files
OUT_ROOT=/home/kan/Research/SignMem/datasets/how2sign/shubert/hands
POSE_ROOT=/home/kan/Research/SignMem/datasets/how2sign/shubert/poses
BATCH_SIZE=200
WORKERS=4

source "$VENV"
export SCRIPT LIST_DIR OUT_ROOT POSE_ROOT BATCH_SIZE

for split in val test train; do
    export SPLIT="$split"
    mkdir -p "$OUT_ROOT/$split"
    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$LIST_DIR/how2sign_crops_$split.list','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] $split: $n videos, $chunks chunks, $WORKERS workers"
    seq 0 $((chunks - 1)) | xargs -P $WORKERS -I{} bash -c '
        python "$SCRIPT" --index {} \
            --batch_size "$BATCH_SIZE" \
            --time_limit 864000 \
            --files_list "$LIST_DIR/how2sign_crops_$SPLIT.list" \
            --pose_path "$POSE_ROOT/$SPLIT" \
            --hand_path "$OUT_ROOT/$SPLIT" \
            --problem_file_path "$LIST_DIR/problems_hands_$SPLIT.txt"
    ' 2>>"$LIST_DIR/hands_$SPLIT.err.log"
done
echo "[$(date)] hands crops done"
