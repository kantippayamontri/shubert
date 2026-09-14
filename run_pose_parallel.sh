#!/bin/bash
set -u

VENV="${VENV:-/home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate}"
SCRIPT="${SCRIPT:-/home/kan/Research/SHuBERT/dataset/kpe_mediapipe.py}"
LIST_DIR="${LIST_DIR:-/home/kan/Research/SHuBERT/how2sign/manifest_files}"
CROPS_ROOT="${CROPS_ROOT:-/home/kan/Research/SignMem/datasets/how2sign/shubert/crops}"
OUT_ROOT="${OUT_ROOT:-/home/kan/Research/SignMem/datasets/how2sign/shubert}"
FACE_MODEL="${FACE_MODEL:-/home/kan/Research/SHuBERT/weights/mediapipe/face_landmarker.task}"
HAND_MODEL="${HAND_MODEL:-/home/kan/Research/SHuBERT/weights/mediapipe/hand_landmarker.task}"
BATCH_SIZE="${BATCH_SIZE:-200}"
WORKERS="${WORKERS:-8}"
SPLITS="${SPLITS:-val test train}"

source "$VENV"
mkdir -p "$LIST_DIR"
export SCRIPT LIST_DIR CROPS_ROOT OUT_ROOT FACE_MODEL HAND_MODEL BATCH_SIZE

for split in $SPLITS; do
    export SPLIT="$split"
    # build a crops .list for this split (absolute paths to cropped videos)
    python -c "
import gzip, pickle, glob
vids = sorted(glob.glob('$CROPS_ROOT/$split/*.mp4'))
with gzip.GzipFile('$LIST_DIR/how2sign_crops_$split.list','wb') as f:
    f.write(pickle.dumps(vids, protocol=0))
print('$split crops list:', len(vids))
"
    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$LIST_DIR/how2sign_crops_$split.list','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] $split: $n videos, $chunks chunks, $WORKERS workers"

    seq 0 $((chunks - 1)) | xargs -P $WORKERS -I{} bash -c '
        python "$SCRIPT" --index {} \
            --batch_size "$BATCH_SIZE" \
            --files_list "$LIST_DIR/how2sign_crops_$SPLIT.list" \
            --pose_path "$OUT_ROOT/poses/$SPLIT" \
            --stats_path "$OUT_ROOT/stats/$SPLIT" \
            --time_limit 864000 \
            --problem_file_path "$LIST_DIR/problems_pose_$SPLIT.txt" \
            --face_model_path "$FACE_MODEL" \
            --hand_model_path "$HAND_MODEL"
    ' 2>>"$LIST_DIR/pose_$SPLIT.err.log"
done
echo "[$(date)] all done"
