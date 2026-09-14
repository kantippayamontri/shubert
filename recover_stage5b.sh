#!/bin/bash
set -euo pipefail

ROOT=/home/kan/Research/SHuBERT
DATA=/mnt/e/datasets/how2sign/shubert
LISTS=$ROOT/how2sign/manifest_files
FE_ENV=$ROOT/.venv-feature-extraction/bin/activate
BATCH=10

source "$FE_ENV"

echo "[$(date)] regenerating 466 face crops"
n=$(python -c "import gzip,pickle; print(len(pickle.load(gzip.open('$LISTS/recovery_faces_train.list','rb'))))")
chunks=$(( (n + BATCH - 1) / BATCH ))
for i in $(seq 0 $((chunks - 1))); do
    python "$ROOT/dataset/crop_face.py" \
        --index "$i" --batch_size "$BATCH" --time_limit 864000 \
        --files_list "$LISTS/recovery_faces_train.list" \
        --pose_path "$DATA/poses/train" \
        --face_path "$DATA/faces/train" \
        --problem_file_path "$LISTS/problems_face_recovery_train.txt"
done

echo "[$(date)] regenerating 95 hand crop pairs"
n=$(python -c "import gzip,pickle; print(len(pickle.load(gzip.open('$LISTS/recovery_hands_train.list','rb'))))")
chunks=$(( (n + BATCH - 1) / BATCH ))
for i in $(seq 0 $((chunks - 1))); do
    python "$ROOT/dataset/crop_hands.py" \
        --index "$i" --batch_size "$BATCH" --time_limit 864000 \
        --files_list "$LISTS/recovery_hands_train.list" \
        --pose_path "$DATA/poses/train" \
        --hand_path "$DATA/hands/train" \
        --problem_file_path "$LISTS/problems_hands_recovery_train.txt"
done

python - <<'PY'
from pathlib import Path
import gzip, pickle
root=Path('/mnt/e/datasets/how2sign/shubert')
lists=Path('/home/kan/Research/SHuBERT/how2sign/manifest_files')
for kind, crop_dir, feat_dir in [('face','faces','face_feats'),('hand','hands','hand_feats')]:
    crops=sorted((root/crop_dir/'train').glob('*.mp4'))
    bad=[p for p in crops if p.stat().st_size == 0]
    if bad:
        raise SystemExit(f'{kind}: {len(bad)} zero-byte crops remain')
    features={p.stem for p in (root/feat_dir/'train').glob('*.npy')}
    missing=[str(p) for p in crops if p.stem not in features]
    with gzip.GzipFile(lists/f'recovery_{kind}_dino_train.list','wb') as f:
        pickle.dump(missing,f,protocol=0)
    print(kind,'missing DINO features:',len(missing))
PY

echo "[$(date)] extracting missing face DINO features"
python "$ROOT/features/dinov2_features.py" \
    --index 0 --batch_size 1000 --time_limit 864000 \
    --files_list "$LISTS/recovery_face_dino_train.list" \
    --output_folder "$DATA/face_feats/train" \
    --dino_path "$ROOT/weights/face_dinov2_checkpoint.pth"

echo "[$(date)] extracting missing hand DINO features"
python "$ROOT/features/dinov2_features.py" \
    --index 0 --batch_size 1000 --time_limit 864000 \
    --files_list "$LISTS/recovery_hand_dino_train.list" \
    --output_folder "$DATA/hand_feats/train" \
    --dino_path "$ROOT/weights/hands_dinov2_checkpoint.pth"

python - <<'PY'
from pathlib import Path
root=Path('/mnt/e/datasets/how2sign/shubert')
checks=[('faces',31047),('hands',62094),('face_feats',31047),('hand_feats',62094)]
for name, expected in checks:
    ext='*.mp4' if name in ('faces','hands') else '*.npy'
    files=list((root/name/'train').glob(ext))
    zero=sum(p.stat().st_size == 0 for p in files)
    print(name,len(files),'zero-byte',zero,'expected',expected)
    if len(files) != expected or zero:
        raise SystemExit(1)
PY

echo "[$(date)] recovery complete"
