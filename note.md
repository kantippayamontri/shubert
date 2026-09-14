# SHuBERT Feature Extraction — How2Sign

## Goal
Extract SHuBERT features `[L=12, T, 768]` from How2Sign (train/val/test).
- L=12 transformer layers, T=frames, 768=embedding dim per frame
- Running locally: single RTX 4080, no SLURM, uv venvs (no conda)

## Pipeline
| Stage | Script | Venv | Status |
|---|---|---|---|
| 1. File lists | `prepare_lists.py` | any | DONE |
| 2. Signer crop | `dataset/clips_bbox.py` | feature-extraction | RUNNING |
| 3. Pose detection | `dataset/kpe_mediapipe.py` | feature-extraction | pending |
| 4. Face/hand crops | `dataset/crop_face.py`, `crop_hands.py` | feature-extraction | pending |
| 5a. Body features | `features/body_features.py` | feature-extraction | pending |
| 5b. Dino features | `features/dinov2_features.py` x2 | dino | pending |
| 6. SHuBERT inference | `features/shubert_inference.py` | shubert | DONE (hidden states) |

## Paths
```
Dataset:   /home/kan/Research/SignMem/datasets/how2sign/{train,val,test}/raw_videos/
Lists:     /home/kan/Research/SHuBERT/how2sign/manifest_files/how2sign_{train,val,test}.list
Outputs:   /home/kan/Research/SignMem/datasets/how2sign/shubert/
Weights:   /home/kan/Research/SHuBERT/weights/
```

Output subfolders (created per stage):
```
shubert/crops/         stage 2
shubert/poses/         stage 3
shubert/faces/         stage 4
shubert/hands/         stage 4
shubert/body_feats/    stage 5a
shubert/face_feats/    stage 5b
shubert/hand_feats/    stage 5b
shubert/shubert_feats/ stage 6 (FFN outputs, legacy)
shubert/shubert_hidden_feats/ stage 6 (hidden states, current)
```

## Assets (`weights/`)
| File | Used by |
|---|---|
| `checkpoint_836_400000.pt` | stage 6 SHuBERT |
| `face_dinov2_checkpoint.pth` | stage 5b face |
| `hands_dinov2_checkpoint.pth` | stage 5b hands |
| `yolov8n.pt` | stage 2 crop |
| `mediapipe/face_landmarker.task` | stage 3 pose |
| `mediapipe/hand_landmarker.task` | stage 3 pose |

## Venvs
```
.venv-feature-extraction  ->  stages 2, 3, 4, 5a
.venv-dino                ->  stage 5b
.venv-shubert             ->  stage 6
```
All `.sh` scripts already patched: `conda activate` replaced with `source .venv-XXX/bin/activate`.

## Data counts
| Split | Videos |
|---|---|
| train | 31,047 |
| val | 1,739 |
| test | 2,343 |

---

## Stage 1 — File lists (DONE)
Generates gzip+pickled list of absolute `*.mp4` paths per split (used by all stage scripts).

```bash
python prepare_lists.py \
    --root /home/kan/Research/SignMem/datasets/how2sign \
    --out_dir /home/kan/Research/SHuBERT/how2sign/manifest_files \
    --splits val,test,train
```

Inspect a list:
```bash
python -c "import gzip,pickle; l=pickle.load(gzip.open('how2sign/manifest_files/how2sign_val.list','rb')); print(len(l), l[0])"
```

---

## Stage 2 — Signer crop (RUNNING)
YOLOv8 crops each video to signer bounding box. Resume-safe (skips existing outputs).

**Run (4 parallel workers, val→test→train):**
```bash
nohup /home/kan/Research/SHuBERT/run_crops_parallel.sh \
    > /home/kan/Research/SHuBERT/how2sign/manifest_files/crops_parallel.log 2>&1 &
```

Underlying command per chunk (what the runner calls internally):
```bash
source /home/kan/Research/SHuBERT/.venv-feature-extraction/bin/activate
python /home/kan/Research/SHuBERT/dataset/clips_bbox.py \
    --index $i \
    --batch_size 100 \
    --files_list /home/kan/Research/SHuBERT/how2sign/manifest_files/how2sign_<split>.list \
    --output_clips_directory /home/kan/Research/SignMem/datasets/how2sign/shubert/crops/<split> \
    --problem_file_path /home/kan/Research/SHuBERT/how2sign/manifest_files/problems_<split>.txt \
    --yolo_model_path /home/kan/Research/SHuBERT/weights/yolov8n.pt
```

Check progress:
```bash
pgrep -f run_crops_parallel          # confirm alive
ls .../shubert/crops/val/*.mp4 | wc -l   # target 1739
ls .../shubert/crops/test/*.mp4 | wc -l  # target 2343
ls .../shubert/crops/train/*.mp4 | wc -l # target 31047
tail how2sign/manifest_files/crops_parallel.log
```

Expected time: val ~25min, test ~45min, train ~9.5h.

---

## Code fixes required before stages 3–4
- **`kpe_mediapipe.py`** lines 152, 253–254: `pose_path` used as `Path` but passed as `str`
  → add `pose_path = Path(args.pose_path)` and `stats_path = Path(args.stats_path)` after argparse
- **`body_features.sh`**: remove stray `--pose_path` flag (script only accepts `--pose_features_path`)

---

## Stage 6 representation change

Stage 6 now extracts **complete transformer hidden states** (`layer[0]`) instead of FFN branch outputs (`layer[-1]`).

- **Hidden states**: post-residual, post-layer-norm transformer states. Standard downstream representation.
- **FFN outputs**: intermediate FFN branch values. Non-standard, kept in `shubert_feats/` for reference.
- **New outputs**: `/mnt/e/datasets/how2sign/shubert/shubert_hidden_feats/`
- **Legacy outputs**: `/mnt/e/datasets/how2sign/shubert/shubert_feats/` (FFN outputs)

Both are `float32 [12,T,768]`. Hidden states differ from FFN outputs (max diff ~178, mean ~0.3).
