#!/bin/bash
set -u

VENV="${VENV:-/home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate}"
SCRIPT="${SCRIPT:-/home/kan/Research/SHuBERT/features/dinov2_features.py}"
LIST_DIR="${LIST_DIR:-/home/kan/Research/SHuBERT/how2sign/manifest_files}"
FACES_ROOT="${FACES_ROOT:-/mnt/e/datasets/how2sign/shubert/faces}"
HANDS_ROOT="${HANDS_ROOT:-/mnt/e/datasets/how2sign/shubert/hands}"
FACE_OUT="${FACE_OUT:-/mnt/e/datasets/how2sign/shubert/face_feats}"
HAND_OUT="${HAND_OUT:-/mnt/e/datasets/how2sign/shubert/hand_feats}"
FACE_DINO="${FACE_DINO:-/home/kan/Research/SHuBERT/weights/face_dinov2_checkpoint.pth}"
HAND_DINO="${HAND_DINO:-/home/kan/Research/SHuBERT/weights/hands_dinov2_checkpoint.pth}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
SPLITS="${SPLITS:-val test train}"

source "$VENV"
mkdir -p "$LIST_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] === FACE DINO FEATS ==="
for split in $SPLITS; do
    mkdir -p "$FACE_OUT/$split"
    python -c "
import gzip, pickle, glob
vids = sorted(glob.glob('$FACES_ROOT/$split/*_face.mp4'))
with gzip.GzipFile('$LIST_DIR/how2sign_faces_$split.list', 'wb') as f:
    f.write(pickle.dumps(vids, protocol=0))
print('$split faces list:', len(vids))
"
    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$LIST_DIR/how2sign_faces_$split.list','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] face DINO $split: $n videos, $chunks chunks"
    for i in $(seq 0 $((chunks - 1))); do
        python "$SCRIPT" --index "$i" \
            --batch_size "$BATCH_SIZE" \
            --time_limit 864000 \
            --files_list "$LIST_DIR/how2sign_faces_$split.list" \
            --output_folder "$FACE_OUT/$split" \
            --dino_path "$FACE_DINO" \
            2>>"$LIST_DIR/dino_face_$split.err.log"
    done
    echo "[$(date)] face DINO $split done"
done

echo "[$(date)] === HANDS DINO FEATS ==="
for split in $SPLITS; do
    mkdir -p "$HAND_OUT/$split"
    python -c "
import gzip, pickle, glob
vids = sorted(glob.glob('$HANDS_ROOT/$split/*_hand[12].mp4'))
with gzip.GzipFile('$LIST_DIR/how2sign_hands_$split.list', 'wb') as f:
    f.write(pickle.dumps(vids, protocol=0))
print('$split hands list:', len(vids))
"
    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$LIST_DIR/how2sign_hands_$split.list','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] hands DINO $split: $n videos, $chunks chunks"
    for i in $(seq 0 $((chunks - 1))); do
        python "$SCRIPT" --index "$i" \
            --batch_size "$BATCH_SIZE" \
            --time_limit 864000 \
            --files_list "$LIST_DIR/how2sign_hands_$split.list" \
            --output_folder "$HAND_OUT/$split" \
            --dino_path "$HAND_DINO" \
            2>>"$LIST_DIR/dino_hands_$split.err.log"
    done
    echo "[$(date)] hands DINO $split done"
done

echo "[$(date)] === STAGE 5b ALL DINO FEATS DONE ==="
