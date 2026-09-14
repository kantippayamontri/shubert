import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("OUT_ROOT", "/mnt/e/datasets/how2sign/shubert"))
MANIFEST_DIR = Path(os.environ.get("LIST_DIR", "/home/kan/Research/SHuBERT/how2sign/manifest_files"))
SPLITS = os.environ.get("SPLITS", "val test train").split()
# Optional: comma-separated split:count pairs, e.g. "val:1739,test:2343,train:31047".
# When unset, expected count is derived from however many face-feature files exist
# for that split (no strict-count check against a fixed dataset size).
EXPECTED_ENV = os.environ.get("EXPECTED", "")
EXPECTED = dict(pair.split(":") for pair in EXPECTED_ENV.split(",") if pair)
EXPECTED = {k: int(v) for k, v in EXPECTED.items()}
STRICT_VALIDATE_SPLITS = set(os.environ.get("STRICT_VALIDATE_SPLITS", "val test").split())


def build_manifest(split: str) -> None:
    rows = []
    mismatches = []
    face_dir = ROOT / "face_feats" / split
    hand_dir = ROOT / "hand_feats" / split
    body_dir = ROOT / "body_feats" / split
    strict = split in STRICT_VALIDATE_SPLITS

    for face_path in sorted(face_dir.glob("*_face.npy")):
        stem = face_path.name.removesuffix("_face.npy")
        paths = (
            face_path,
            hand_dir / f"{stem}_hand1.npy",
            hand_dir / f"{stem}_hand2.npy",
            body_dir / f"{stem}_pose.npy",
        )
        if strict and not all(path.exists() and path.stat().st_size > 0 for path in paths):
            missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
            raise RuntimeError(f"{split}/{stem}: missing or empty streams: {missing}")
        if strict:
            lengths = [np.load(path, mmap_mode="r").shape[0] for path in paths]
            if len(set(lengths)) != 1:
                mismatches.append((stem, lengths))
                continue
        rows.append("\t".join(map(str, paths)))

    if mismatches:
        preview = "\n".join(f"{stem}: {lengths}" for stem, lengths in mismatches[:10])
        raise RuntimeError(f"{split}: {len(mismatches)} length mismatches\n{preview}")
    if split in EXPECTED and len(rows) != EXPECTED[split]:
        raise RuntimeError(f"{split}: expected {EXPECTED[split]} complete rows, got {len(rows)}")

    output = MANIFEST_DIR / f"shubert_input_{split}.tsv"
    output.write_text("\n".join(rows) + "\n")
    print(f"{split}: {len(rows)} rows -> {output}")


for split_name in SPLITS:
    build_manifest(split_name)
