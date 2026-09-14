import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeVar

import numpy as np
import torch

from examples.shubert.models.shubert import SHubertModel, SHubertConfig


T = TypeVar("T")


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    face: Path
    left_hand: Path
    right_hand: Path
    body: Path


def read_manifest(path: Path) -> list[SamplePaths]:
    samples = []
    sample_ids = set()
    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                raise ValueError(f"Line {line_num}: expected 4 columns, got {len(parts)}")
            face_path = Path(parts[0])
            if not face_path.name.endswith("_face.npy"):
                raise ValueError(f"Line {line_num}: face path must end with _face.npy: {face_path}")
            sample_id = face_path.name.removesuffix("_face.npy")
            if sample_id in sample_ids:
                raise ValueError(f"Line {line_num}: duplicate sample ID: {sample_id}")
            sample_ids.add(sample_id)
            samples.append(
                SamplePaths(
                    sample_id=sample_id,
                    face=face_path,
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


def load_validated_source(
    sample: SamplePaths, device: torch.device
) -> tuple[list[dict[str, torch.Tensor]], int]:
    for name, path in [
        ("face", sample.face),
        ("left_hand", sample.left_hand),
        ("right_hand", sample.right_hand),
        ("body", sample.body),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    face_np = np.load(sample.face, allow_pickle=False)
    left_np = np.load(sample.left_hand, allow_pickle=False)
    right_np = np.load(sample.right_hand, allow_pickle=False)
    body_np = np.load(sample.body, allow_pickle=False)

    expected_dims = {"face": 384, "left_hand": 384, "right_hand": 384, "body": 14}
    arrays = {
        "face": face_np,
        "left_hand": left_np,
        "right_hand": right_np,
        "body": body_np,
    }
    for name, arr in arrays.items():
        if arr.size == 0:
            raise ValueError(f"{name} is empty")
        if arr.ndim != 2 or arr.shape[1] != expected_dims[name]:
            raise ValueError(f"{name} shape {arr.shape} != [T,{expected_dims[name]}]")
        if np.issubdtype(arr.dtype, np.complexfloating):
            raise ValueError(f"{name} dtype {arr.dtype} is not real-valued")
        if not (
            np.issubdtype(arr.dtype, np.integer)
            or np.issubdtype(arr.dtype, np.floating)
        ):
            raise ValueError(f"{name} dtype {arr.dtype} is not numeric")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")

    converted_arrays = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for name, arr in arrays.items():
            converted = np.ascontiguousarray(arr, dtype=np.float32)
            if not np.isfinite(converted).all():
                raise ValueError(f"{name} contains non-finite values after float32 conversion")
            converted_arrays[name] = converted

    T_face = face_np.shape[0]
    T_left = left_np.shape[0]
    T_right = right_np.shape[0]
    T_body = body_np.shape[0]
    if not (T_face == T_left == T_right == T_body):
        raise ValueError(f"T mismatch: face={T_face}, left={T_left}, right={T_right}, body={T_body}")
    if T_face <= 0:
        raise ValueError("T must be positive")

    face = torch.from_numpy(converted_arrays["face"]).to(device)
    left = torch.from_numpy(converted_arrays["left_hand"]).to(device)
    right = torch.from_numpy(converted_arrays["right_hand"]).to(device)
    body = torch.from_numpy(converted_arrays["body"]).to(device)

    length = T_face
    source = [{
        "face": face,
        "left_hand": left,
        "right_hand": right,
        "body_posture": body,
        "label_face": torch.zeros((length, 1), device=device),
        "label_left_hand": torch.zeros((length, 1), device=device),
        "label_right_hand": torch.zeros((length, 1), device=device),
        "label_body_posture": torch.zeros((length, 1), device=device),
    }]
    return source, length

def load_model(checkpoint_path: Path, device: torch.device) -> SHubertModel:
    cfg = SHubertConfig()
    model = SHubertModel.build_model(cfg, task=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def validate_feature_array(features: np.ndarray, expected_t: int) -> None:
    if features.ndim != 3:
        raise ValueError(f"ndim {features.ndim} != 3")
    if features.shape != (12, expected_t, 768):
        raise ValueError(f"shape {features.shape} != (12, {expected_t}, 768)")
    if features.dtype != np.float32:
        raise ValueError(f"dtype {features.dtype} != float32")
    if not np.isfinite(features).all():
        raise ValueError("non-finite values")


def output_validation_error(path: Path, expected_t: int) -> str | None:
    if not path.exists():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    try:
        features = np.load(path, allow_pickle=False)
        validate_feature_array(features, expected_t)
    except Exception as e:
        return str(e)
    return None


def atomic_save(path: Path, features: np.ndarray, expected_t: int) -> None:
    validate_feature_array(features, expected_t)
    tmp_path = path.parent / f"{path.stem}.tmp.npy"
    try:
        np.save(tmp_path, features)
        loaded = np.load(tmp_path, allow_pickle=False)
        validate_feature_array(loaded, expected_t)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def extract_hidden_states(model, source, length: int) -> np.ndarray:
    with torch.inference_mode():
        result = model.extract_features(source, padding_mask=None, kmeans_labels=None, mask=False)

    hidden_states = []
    for layer_result in result["layer_results"]:
        hidden = layer_result[0]
        hidden = hidden.squeeze(1)
        hidden_states.append(hidden.cpu().numpy())

    features = np.stack(hidden_states, axis=0).astype(np.float32, copy=False)
    if features.shape != (12, length, 768):
        raise ValueError(f"output shape {features.shape} != (12, {length}, 768)")
    return features


def write_failure(report: TextIO | None, sample: SamplePaths, error: Exception) -> None:
    if report is None:
        return
    failure = {
        "sample_id": sample.sample_id,
        "inputs": [str(sample.face), str(sample.left_hand), str(sample.right_hand), str(sample.body)],
        "error_type": type(error).__name__,
        "error": str(error),
    }
    report.write(json.dumps(failure) + "\n")
    report.flush()


def main(
    samples: list[SamplePaths],
    checkpoint_path: Path,
    output_dir: Path,
    failure_report: Path | None,
    device: torch.device,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    report = None
    if failure_report is not None:
        failure_report.parent.mkdir(parents=True, exist_ok=True)
        report = failure_report.open("w")

    processed = 0
    resumed = 0
    written = 0
    failed = 0
    try:
        model = load_model(checkpoint_path, device)
        for sample in samples:
            processed += 1
            output_path = output_dir / f"{sample.sample_id}.npy"

            try:
                source, length = load_validated_source(sample, device)
                if output_validation_error(output_path, expected_t=length) is None:
                    resumed += 1
                else:
                    features = extract_hidden_states(model, source, length)
                    atomic_save(output_path, features, expected_t=length)
                    written += 1
            except Exception as error:
                failed += 1
                write_failure(report, sample, error)

            if processed % 100 == 0:
                print(f"processed={processed} resumed={resumed} written={written} failed={failed}")
    finally:
        if report is not None:
            report.close()

    print(f"FINAL: processed={processed} resumed={resumed} written={written} failed={failed}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="manifest TSV path")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("--failure_report", type=str, default=None, help="failure report JSONL path")
    parser.add_argument("--index", type=int, default=None, help="chunk index (requires --batch_size)")
    parser.add_argument("--batch_size", type=int, default=None, help="chunk size (requires --index)")
    args = parser.parse_args()

    if (args.index is None) != (args.batch_size is None):
        parser.error("--index and --batch_size must be used together")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = read_manifest(Path(args.csv_path))
    failure_report = Path(args.failure_report) if args.failure_report else None
    if failure_report is not None:
        failure_report.parent.mkdir(parents=True, exist_ok=True)
        failure_report.write_text("")

    if args.index is not None and args.batch_size is not None:
        try:
            samples = select_chunk(samples, args.index, args.batch_size)
        except ValueError as error:
            parser.error(str(error))
        if not samples:
            print(f"chunk {args.index} empty, nothing to do")
            sys.exit(0)

    if not samples:
        print("no samples to process")
        sys.exit(0)

    exit_code = main(samples, Path(args.checkpoint_path), Path(args.output_dir),
                     failure_report, device)
    sys.exit(exit_code)
