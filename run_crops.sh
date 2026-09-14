#!/bin/bash
set -u

VENV=/home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate
SCRIPT=/home/kan/Research/SHuBERT/dataset/clips_bbox.py
LIST_DIR=/home/kan/Research/SHuBERT/how2sign/manifest_files
OUT_ROOT=/home/kan/Research/SignMem/datasets/how2sign/shubert/crops
YOLO=/home/kan/Research/SHuBERT/weights/yolov8n.pt
BATCH_SIZE=100

source "$VENV"

for split in val test; do
    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$LIST_DIR/how2sign_$split.list','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] $split: $n videos, $chunks chunks"
    mkdir -p "$OUT_ROOT/$split"
    for i in $(seq 0 $((chunks - 1))); do
        echo "[$(date)] $split chunk $i/$((chunks-1))"
        python "$SCRIPT" --index "$i" \
            --batch_size "$BATCH_SIZE" \
            --files_list "$LIST_DIR/how2sign_$split.list" \
            --output_clips_directory "$OUT_ROOT/$split" \
            --problem_file_path "$LIST_DIR/problems_$split.txt" \
            --yolo_model_path "$YOLO" \
            2>>"$LIST_DIR/crops_$split.err.log"
    done
done
echo "[$(date)] all done"
