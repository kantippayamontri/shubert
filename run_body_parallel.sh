#!/bin/bash
set -u

VENV="${VENV:-/home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate}"
SCRIPT="${SCRIPT:-/home/kan/Research/SHuBERT/features/body_features.py}"
LIST_DIR="${LIST_DIR:-/home/kan/Research/SHuBERT/how2sign/manifest_files}"
POSE_ROOT="${POSE_ROOT:-/mnt/e/datasets/how2sign/shubert/poses}"
OUT_ROOT="${OUT_ROOT:-/mnt/e/datasets/how2sign/shubert/body_feats}"
BATCH_SIZE="${BATCH_SIZE:-500}"
WORKERS="${WORKERS:-4}"
SPLITS="${SPLITS:-val test train}"

source "$VENV"
mkdir -p "$LIST_DIR"
export SCRIPT LIST_DIR OUT_ROOT BATCH_SIZE

for split in $SPLITS; do
    export SPLIT="$split"
    mkdir -p "$OUT_ROOT/$split"
    # rebuild the pose-json list for this split (absolute paths under $POSE_ROOT)
    python -c "
import gzip, pickle, glob
vids = sorted(glob.glob('$POSE_ROOT/$split/*_pose.json'))
with gzip.GzipFile('$LIST_DIR/how2sign_poses_$split.list', 'wb') as f:
    f.write(pickle.dumps(vids, protocol=0))
print('$split poses list:', len(vids))
"
    n=$(python -c "import gzip,pickle;print(len(pickle.load(gzip.open('$LIST_DIR/how2sign_poses_$split.list','rb'))))")
    chunks=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[$(date)] body_feats $split: $n videos, $chunks chunks, $WORKERS workers"
    seq 0 $((chunks - 1)) | xargs -P $WORKERS -I{} bash -c '
        python "$SCRIPT" --index {} \
            --batch_size "$BATCH_SIZE" \
            --time_limit 864000 \
            --files_list "$LIST_DIR/how2sign_poses_$SPLIT.list" \
            --pose_features_path "$OUT_ROOT/$SPLIT"
    ' 2>>"$LIST_DIR/body_$SPLIT.err.log"
    echo "[$(date)] body_feats $split done"
done
echo "[$(date)] === STAGE 5a BODY FEATS ALL DONE ==="
