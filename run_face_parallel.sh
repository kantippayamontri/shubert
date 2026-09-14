#!/bin/bash
set -u

VENV=/home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate
SCRIPT=/home/kan/Research/SHuBERT/dataset/crop_face.py
LIST_DIR=/home/kan/Research/SHuBERT/how2sign/manifest_files
CROPS_ROOT=/home/kan/Research/SignMem/datasets/how2sign/shubert/crops
OUT_ROOT=/home/kan/Research/SignMem/datasets/how2sign/shubert/faces
POSE_ROOT=/home/kan/Research/SignMem/datasets/how2sign/shubert/poses
BATCH_SIZE=200
WORKERS=8

source "$VENV"
export SCRIPT LIST_DIR CROPS_ROOT OUT_ROOT POSE_ROOT BATCH_SIZE

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
            --face_path "$OUT_ROOT/$SPLIT" \
            --problem_file_path "$LIST_DIR/problems_face_$SPLIT.txt"
    ' 2>>"$LIST_DIR/face_$SPLIT.err.log"
done
echo "[$(date)] face crops done"
