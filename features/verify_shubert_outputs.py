#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypeVar

import numpy as np


# Optional: comma-separated split:count pairs, e.g. "val:1739,test:2343,train:31047".
# When a split has no entry, --split for that split skips the exact-count check.
_EXPECTED_ENV = os.environ.get("EXPECTED", "")
EXPECTED_COUNTS = {k: int(v) for k, v in (pair.split(":") for pair in _EXPECTED_ENV.split(",") if pair)}
T = TypeVar("T")


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    face: Path
    left_hand: Path
    right_hand: Path
    body: Path


@dataclass(frozen=True)
class VerificationResult:
    valid: int
    invalid_files: list[str]
    missing: list[str]
    unexpected: list[str]
    duplicates: list[str]

    @property
    def invalid(self) -> int:
        return len(self.invalid_files)

    @property
    def ok(self) -> bool:
        return not (self.invalid_files or self.missing or self.unexpected or self.duplicates)


def read_manifest(path: Path) -> list[ManifestSample]:
    samples = []
    sample_ids = set()
    with path.open() as manifest:
        for line_number, line in enumerate(manifest, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Line {line_number}: expected 4 columns, got {len(parts)}")
            face = Path(parts[0])
            if not face.name.endswith("_face.npy"):
                raise ValueError(f"Line {line_number}: face path must end with _face.npy: {face}")
            sample_id = face.name.removesuffix("_face.npy")
            if sample_id in sample_ids:
                raise ValueError(f"Line {line_number}: duplicate sample ID: {sample_id}")
            sample_ids.add(sample_id)
            samples.append(
                ManifestSample(
                    sample_id=sample_id,
                    face=face,
                    left_hand=Path(parts[1]),
                    right_hand=Path(parts[2]),
                    body=Path(parts[3]),
                )
            )
    return samples


def select_chunk(items: list[T], index: int, batch_size: int) -> list[T]:
    if index < 0:
        raise ValueError("index must be nonnegative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = index * batch_size
    return items[start : start + batch_size]


def validate_array(path: Path, expected_width: int) -> int:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(f"{path.name}: shape {array.shape} != [T,{expected_width}]")
    if array.shape[0] <= 0:
        raise ValueError(f"{path.name}: empty")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{path.name}: dtype {array.dtype} is not real-valued")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"{path.name}: dtype {array.dtype} is not numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: non-finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"{path.name}: non-finite values after float32 conversion")
    return array.shape[0]


def validate_sample_inputs(sample: ManifestSample) -> int:
    lengths = [
        validate_array(sample.face, 384),
        validate_array(sample.left_hand, 384),
        validate_array(sample.right_hand, 384),
        validate_array(sample.body, 14),
    ]
    if len(set(lengths)) != 1:
        raise ValueError(f"input T mismatch: {lengths}")
    return lengths[0]


def validate_output(path: Path, expected_t: int) -> None:
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    if features.shape != (12, expected_t, 768):
        raise ValueError(f"shape {features.shape} != (12, {expected_t}, 768)")
    if features.dtype != np.float32:
        raise ValueError(f"dtype {features.dtype} != float32")
    if not np.isfinite(features).all():
        raise ValueError("non-finite values")


def verify_outputs(
    output_dir: Path,
    samples: list[ManifestSample],
    allowed_output_names: set[str] | None = None,
) -> VerificationResult:
    counts = {}
    samples_by_id = {}
    for sample in samples:
        counts[sample.sample_id] = counts.get(sample.sample_id, 0) + 1
        samples_by_id.setdefault(sample.sample_id, sample)

    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    expected = {f"{sample_id}.npy" for sample_id in samples_by_id}
    allowed = expected if allowed_output_names is None else allowed_output_names
    actual = {path.name for path in output_dir.glob("*.npy")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - allowed)
    invalid_files = []
    valid = 0

    for sample_id, sample in samples_by_id.items():
        output_name = f"{sample_id}.npy"
        if output_name in missing:
            continue
        try:
            length = validate_sample_inputs(sample)
            validate_output(output_dir / output_name, length)
            valid += 1
        except Exception as error:
            invalid_files.append(f"{output_name}: {error}")

    return VerificationResult(
        valid=valid,
        invalid_files=invalid_files,
        missing=missing,
        unexpected=unexpected,
        duplicates=duplicates,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHuBERT hidden-state outputs")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest TSV path")
    parser.add_argument("--output_dir", type=Path, required=True, help="output directory")
    parser.add_argument("--split", choices=["val", "test", "train"], help="split name")
    parser.add_argument("--index", type=int, default=None, help="chunk index (requires --batch_size)")
    parser.add_argument("--batch_size", type=int, default=None, help="chunk size (requires --index)")
    args = parser.parse_args()

    if (args.index is None) != (args.batch_size is None):
        parser.error("--index and --batch_size must be used together")
    if args.split and args.index is not None:
        parser.error("--split cannot be combined with chunk verification")
    if not args.output_dir.exists():
        parser.error(f"output directory does not exist: {args.output_dir}")

    all_samples = read_manifest(args.manifest)
    expected_count = EXPECTED_COUNTS.get(args.split) if args.split else None
    if expected_count is not None and len(all_samples) != expected_count:
        print(f"ERROR: expected {expected_count} samples for {args.split}, got {len(all_samples)}")
        return 1

    allowed_output_names = {f"{sample.sample_id}.npy" for sample in all_samples}
    samples = all_samples
    if args.index is not None:
        assert args.batch_size is not None
        try:
            samples = select_chunk(samples, args.index, args.batch_size)
        except ValueError as error:
            parser.error(str(error))
        if not samples:
            print(f"chunk {args.index} empty, nothing to verify")
            return 0

    result = verify_outputs(args.output_dir, samples, allowed_output_names)
    for label, values in (
        ("INVALID", result.invalid_files),
        ("MISSING", result.missing),
        ("UNEXPECTED", result.unexpected),
        ("DUPLICATE", result.duplicates),
    ):
        for value in values:
            print(f"{label}: {value}")

    print(
        f"RESULTS: valid={result.valid} invalid={result.invalid} "
        f"missing={len(result.missing)} unexpected={len(result.unexpected)} "
        f"duplicates={len(result.duplicates)}"
    )
    if not result.ok:
        return 1
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
