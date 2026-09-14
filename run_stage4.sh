#!/bin/bash
set -u

VENV="${VENV:-/home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate}"
LIST_DIR="${LIST_DIR:-/home/kan/Research/SHuBERT/how2sign/manifest_files}"
CROPS_ROOT="${CROPS_ROOT:-/mnt/e/datasets/how2sign/shubert/crops}"
POSE_ROOT="${POSE_ROOT:-/mnt/e/datasets/how2sign/shubert/poses}"
FACES_ROOT="${FACES_ROOT:-/mnt/e/datasets/how2sign/shubert/faces}"
HANDS_ROOT="${HANDS_ROOT:-/mnt/e/datasets/how2sign/shubert/hands}"
FACE_SCRIPT="${FACE_SCRIPT:-/home/kan/Research/SHuBERT/dataset/crop_face.py}"
HAND_SCRIPT="${HAND_SCRIPT:-/home/kan/Research/SHuBERT/dataset/crop_hands.py}"
BATCH_SIZE="${BATCH_SIZE:-50}"
WORKERS="${WORKERS:-2}"
SPLITS="${SPLITS:-val test train}"

source "$VENV"
mkdir -p "$LIST_DIR"
export LIST_DIR CROPS_ROOT POSE_ROOT FACES_ROOT HANDS_ROOT FACE_SCRIPT HAND_SCRIPT BATCH_SIZE

echo "[$(date)] === FACES ==="
for split in $SPLITS; do
    export SPLIT="$split"
    mkdir -p "$FACES_ROOT/$split"
    if [ ! -f "$LIST_DIR/how2sign_crops_$split.list" ]; then
        echo "ERROR: $LIST_DIR/how2sign_crops_$split.list missing. Run run_pose_parallel.sh first (Stage 3)." >&2
        exit 1
    fi
    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$LIST_DIR/how2sign_crops_$split.list','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] faces $split: $n videos, $chunks chunks, $WORKERS workers"
    seq 0 $((chunks - 1)) | xargs -P $WORKERS -I{} bash -c '
        python "$FACE_SCRIPT" --index {} \
            --batch_size "$BATCH_SIZE" \
            --time_limit 864000 \
            --files_list "$LIST_DIR/how2sign_crops_$SPLIT.list" \
            --pose_path "$POSE_ROOT/$SPLIT" \
            --face_path "$FACES_ROOT/$SPLIT" \
            --problem_file_path "$LIST_DIR/problems_face_$SPLIT.txt"
    ' 2>>"$LIST_DIR/face_$SPLIT.err.log"
    echo "[$(date)] faces $split done"
done

echo "[$(date)] === HANDS ==="
for split in $SPLITS; do
    export SPLIT="$split"
    mkdir -p "$HANDS_ROOT/$split"
    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$LIST_DIR/how2sign_crops_$split.list','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] hands $split: $n videos, $chunks chunks, $WORKERS workers"
    seq 0 $((chunks - 1)) | xargs -P $WORKERS -I{} bash -c '
        python "$HAND_SCRIPT" --index {} \
            --batch_size "$BATCH_SIZE" \
            --time_limit 864000 \
            --files_list "$LIST_DIR/how2sign_crops_$SPLIT.list" \
            --pose_path "$POSE_ROOT/$SPLIT" \
            --hand_path "$HANDS_ROOT/$SPLIT" \
            --problem_file_path "$LIST_DIR/problems_hands_$SPLIT.txt"
    ' 2>>"$LIST_DIR/hands_$SPLIT.err.log"
    echo "[$(date)] hands $split done"
done

echo "[$(date)] === STAGE 4 ALL DONE ==="
