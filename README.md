
## SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction [ACL 2025]

This repository contains research code for the paper [*SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction*](https://arxiv.org/abs/2411.16765).

<p align="middle">
  <img src="imgs/shubert_pipeline.png"  alt="SHuBERT Overview">
</p>


We introduce SHuBERT (Sign Hidden-Unit BERT), a self-supervised contextual representation model learned from approximately 1,000 hours of  American Sign Language video. SHuBERT adapts masked token prediction objectives to multi-stream visual sign language input, learning to predict multiple targets corresponding to clustered hand, face, and body pose streams. SHuBERT achieves state-of-the-art performance across multiple tasks including sign language translation, isolated sign language recognition, and fingerspelling detection.  

----

### Installation

We provide installation and inference instructions in [QUICKSTART.md](QUICKSTART.md).

### Usage
#### 1. Preparing the data

We describe how to prepare the datasets in [DATASETS.md](DATASETS.md).


#### 2. Model Weights

Please download the weight of SHuBERT (as well as the DINO Face and Hand) weights [link](https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV?usp=sharing).

#### 3. Feature Extraction

We describe how to extract features from the pretrained model in [FEATURES.md](FEATURES.md).

##### Running the full pipeline on a custom dataset

The full pipeline (signer crop → pose landmarks → face/hand crops → body/DINOv2
features → SHuBERT hidden states) is orchestrated by the `run_*.sh` scripts.
Every script reads its paths from environment variables, so a separate dataset
and output tree can be processed without editing the scripts:

```bash
DATASET_ROOT=/mnt/e/how2sign_sentences
LIST_DIR=/home/kan/Research/SHuBERT/how2sign_realign/manifest_files
OUT_ROOT_BASE=/mnt/e/how2sign_sentences/shubert

# Stage 2: signer crop (expects $DATASET_ROOT/<split>_sentence_videos/*.mp4)
env DATASET_ROOT="$DATASET_ROOT" LIST_DIR="$LIST_DIR" OUT_ROOT="$OUT_ROOT_BASE/crops" \
    bash run_crops_parallel.sh

# Stage 3: pose / face / hand landmarks
env LIST_DIR="$LIST_DIR" CROPS_ROOT="$OUT_ROOT_BASE/crops" OUT_ROOT="$OUT_ROOT_BASE" \
    bash run_pose_parallel.sh

# Stage 4: face and hand crop videos
env LIST_DIR="$LIST_DIR" CROPS_ROOT="$OUT_ROOT_BASE/crops" POSE_ROOT="$OUT_ROOT_BASE/poses" \
    FACES_ROOT="$OUT_ROOT_BASE/faces" HANDS_ROOT="$OUT_ROOT_BASE/hands" \
    bash run_stage4.sh

# Stage 5a: body pose features
env LIST_DIR="$LIST_DIR" POSE_ROOT="$OUT_ROOT_BASE/poses" OUT_ROOT="$OUT_ROOT_BASE/body_feats" \
    bash run_body_parallel.sh

# Stage 5b: DINOv2 face / hand features
env LIST_DIR="$LIST_DIR" FACES_ROOT="$OUT_ROOT_BASE/faces" HANDS_ROOT="$OUT_ROOT_BASE/hands" \
    FACE_OUT="$OUT_ROOT_BASE/face_feats" HAND_OUT="$OUT_ROOT_BASE/hand_feats" \
    bash run_dino_all.sh

# Stage 6a: build the four-stream manifest
env OUT_ROOT="$OUT_ROOT_BASE" LIST_DIR="$LIST_DIR" \
    python build_shubert_manifests.py

# Stage 6b: SHuBERT hidden-state extraction + verification
env LIST_DIR="$LIST_DIR" OUTPUT="$OUT_ROOT_BASE/shubert_hidden_feats" \
    bash run_shubert.sh
```

Notes:

- `DATASET_ROOT` must contain `<split>_sentence_videos/*.mp4` (e.g.
  `val_sentence_videos`, `test_sentence_videos`, `train_sentence_videos`).
  Override a single split's video directory with `VIDEO_DIR_VAL`, `VIDEO_DIR_TEST`,
  or `VIDEO_DIR_TRAIN`.
- Every stage is resume-safe: existing outputs are skipped on re-run.
- Prefix any stage command with `SPLITS=val` to process a single split, e.g. for a
  quick smoke test before launching the full dataset.
- Each stage builds its input `.list` from the previous stage's output directory,
  so the `LIST_DIR` tree is generated automatically.
- Scripts default to the repository's virtual environments. Override with
  `VENV=/path/to/venv/bin/activate` if your environments live elsewhere.
- Stage 6 writes `float32 [12, T, 768]` transformer hidden states to
  `$OUT_ROOT_BASE/shubert_hidden_feats/<split>/` and verifies every output.
- Optionally set `EXPECTED=val:1741,test:2357,train:31165` (adjust to your
  dataset) to enforce exact per-split sample counts in manifest building and
  verification. When unset, counts are not checked.


#### 4. Pretraining

- sbatch train_shubert.sh


#### 5. Fine-tuning on Downstream Tasks

TODO

---- 
### Citing our work
If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{gueuwou-etal-2025-shubert,
    title = "SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction",
    author = "Gueuwou, Shester and Du, Xiaodan and Shakhnarovich, Greg and Livescu, Karen and Liu, Alexander H.",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
}
```


### References
This codebase is heavily influenced by the [DinoSR](https://github.com/Alexander-H-Liu/dinosr) and [Fairseq](https://github.com/facebookresearch/fairseq) repositories.

### License
This project is primarily under the MIT license.