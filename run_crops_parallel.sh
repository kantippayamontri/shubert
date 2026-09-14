#!/bin/bash
set -u

VENV="${VENV:-/home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate}"
SCRIPT="${SCRIPT:-/home/kan/Research/SHuBERT/dataset/clips_bbox.py}"
LIST_DIR="${LIST_DIR:-/home/kan/Research/SHuBERT/how2sign/manifest_files}"
OUT_ROOT="${OUT_ROOT:-/home/kan/Research/SignMem/datasets/how2sign/shubert/crops}"
YOLO="${YOLO:-/home/kan/Research/SHuBERT/weights/yolov8n.pt}"
BATCH_SIZE="${BATCH_SIZE:-100}"
WORKERS="${WORKERS:-4}"
# Optional: source video root. If set, video_dir_<split> is derived as
# $DATASET_ROOT/<split>_sentence_videos (override per-split with e.g. VIDEO_DIR_VAL).
DATASET_ROOT="${DATASET_ROOT:-}"
SPLITS="${SPLITS:-val test train}"

source "$VENV"
mkdir -p "$LIST_DIR"
export SCRIPT LIST_DIR OUT_ROOT YOLO BATCH_SIZE

for split in $SPLITS; do
    mkdir -p "$OUT_ROOT/$split"
    export SPLIT="$split"

    list_file="$LIST_DIR/how2sign_$split.list"
    if [ ! -f "$list_file" ]; then
        if [ -z "$DATASET_ROOT" ]; then
            echo "ERROR: $list_file does not exist and DATASET_ROOT is not set." >&2
            echo "Set DATASET_ROOT=/path/to/videos (expects \$DATASET_ROOT/${split}_sentence_videos/*.mp4)" >&2
            exit 1
        fi
        video_dir_var="VIDEO_DIR_$(echo "$split" | tr '[:lower:]' '[:upper:]')"
        video_dir="${!video_dir_var:-$DATASET_ROOT/${split}_sentence_videos}"
        echo "[$(date)] building $list_file from $video_dir"
        python -c "
import gzip, pickle, glob
vids = sorted(glob.glob('$video_dir/*.mp4'))
with gzip.GzipFile('$list_file', 'wb') as f:
    f.write(pickle.dumps(vids, protocol=0))
print('$split source list:', len(vids))
"
    fi

    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$list_file','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] $SPLIT: $n videos, $chunks chunks, $WORKERS workers"
    seq 0 $((chunks - 1)) | xargs -P $WORKERS -I{} bash -c '
        python "$SCRIPT" --index {} \
            --batch_size "$BATCH_SIZE" \
            --files_list "$LIST_DIR/how2sign_$SPLIT.list" \
            --output_clips_directory "$OUT_ROOT/$SPLIT" \
            --problem_file_path "$LIST_DIR/problems_$SPLIT.txt" \
            --yolo_model_path "$YOLO"
    ' 2>>"$LIST_DIR/crops_$SPLIT.err.log"
done
echo "[$(date)] all done"
