import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from features.shubert_inference import (
    SamplePaths,
    load_validated_source,
    main,
    read_manifest,
    select_chunk,
)


class FakeModel:
    def extract_features(self, source, **kwargs):
        length = source[0]["face"].shape[0]
        layers = []
        for index in range(12):
            hidden = torch.full((length, 1, 768), float(index))
            ffn = torch.full((length, 1, 768), float(index + 100))
            layers.append((hidden, None, ffn))
        return {"layer_results": layers}


class TestReadManifest(unittest.TestCase):
    def test_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            face = tmpdir / "sample_face.npy"
            left = tmpdir / "sample_hand1.npy"
            right = tmpdir / "sample_hand2.npy"
            body = tmpdir / "sample_pose.npy"
            for p in [face, left, right, body]:
                p.touch()

            manifest = tmpdir / "manifest.tsv"
            manifest.write_text(f"{face}\t{left}\t{right}\t{body}\n")

            samples = read_manifest(manifest)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "sample")
            self.assertEqual(samples[0].face, face)

    def test_wrong_column_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest = tmpdir / "manifest.tsv"
            manifest.write_text("a\tb\tc\n")

            with self.assertRaises(ValueError) as ctx:
                read_manifest(manifest)
            self.assertIn("4 columns", str(ctx.exception))

    def test_duplicate_sample_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = "\t".join(
                str(root / name)
                for name in ("sample_face.npy", "sample_hand1.npy", "sample_hand2.npy", "sample_pose.npy")
            )
            manifest = root / "manifest.tsv"
            manifest.write_text(f"{row}\n{row}\n")

            with self.assertRaisesRegex(ValueError, "duplicate sample ID"):
                read_manifest(manifest)

    def test_face_suffix_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.tsv"
            manifest.write_text(
                "\t".join(
                    str(root / name)
                    for name in ("sample.npy", "sample_hand1.npy", "sample_hand2.npy", "sample_pose.npy")
                )
                + "\n"
            )

            with self.assertRaisesRegex(ValueError, "_face.npy"):
                read_manifest(manifest)


class TestLoadValidatedSource(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _make_sample(self, T=10):
        face = self.tmpdir / "test_face.npy"
        left = self.tmpdir / "test_hand1.npy"
        right = self.tmpdir / "test_hand2.npy"
        body = self.tmpdir / "test_pose.npy"

        np.save(face, np.random.randn(T, 384).astype(np.float32))
        np.save(left, np.random.randn(T, 384).astype(np.float32))
        np.save(right, np.random.randn(T, 384).astype(np.float32))
        np.save(body, np.random.randn(T, 14).astype(np.float64))

        return SamplePaths("test", face, left, right, body)

    def test_valid_source(self):
        sample = self._make_sample(T=10)
        source, length = load_validated_source(sample, self.device)

        self.assertEqual(length, 10)
        self.assertEqual(len(source), 1)
        self.assertEqual(source[0]["face"].shape, (10, 384))
        self.assertEqual(source[0]["left_hand"].shape, (10, 384))
        self.assertEqual(source[0]["right_hand"].shape, (10, 384))
        self.assertEqual(source[0]["body_posture"].shape, (10, 14))
        self.assertEqual(source[0]["face"].dtype, torch.float32)
        self.assertEqual(source[0]["body_posture"].dtype, torch.float32)

    def test_missing_file(self):
        sample = self._make_sample()
        sample.face.unlink()

        with self.assertRaises(FileNotFoundError):
            load_validated_source(sample, self.device)

    def test_wrong_face_shape(self):
        sample = self._make_sample()
        np.save(sample.face, np.random.randn(5, 384).astype(np.float32))

        with self.assertRaises(ValueError) as ctx:
            load_validated_source(sample, self.device)
        self.assertIn("T mismatch", str(ctx.exception))

    def test_scalar_face_rejected_as_wrong_shape(self):
        sample = self._make_sample()
        np.save(sample.face, np.array(1.0, dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "face shape"):
            load_validated_source(sample, self.device)

    def test_wrong_hand_dim(self):
        sample = self._make_sample()
        np.save(sample.left_hand, np.random.randn(10, 512).astype(np.float32))

        with self.assertRaises(ValueError) as ctx:
            load_validated_source(sample, self.device)
        self.assertIn("hand", str(ctx.exception))

    def test_wrong_body_dim(self):
        sample = self._make_sample()
        np.save(sample.body, np.random.randn(10, 7).astype(np.float32))

        with self.assertRaises(ValueError) as ctx:
            load_validated_source(sample, self.device)
        self.assertIn("body", str(ctx.exception))

    def test_nan_in_input(self):
        sample = self._make_sample()
        arr = np.random.randn(10, 384).astype(np.float32)
        arr[0, 0] = np.nan
        np.save(sample.face, arr)

        with self.assertRaises(ValueError) as ctx:
            load_validated_source(sample, self.device)
        self.assertIn("non-finite", str(ctx.exception))

    def test_non_numeric_input(self):
        sample = self._make_sample()
        np.save(sample.face, np.full((10, 384), "bad"))

        with self.assertRaisesRegex(ValueError, "numeric"):
            load_validated_source(sample, self.device)

    def test_complex_input(self):
        sample = self._make_sample()
        np.save(sample.face, np.ones((10, 384), dtype=np.complex64))

        with self.assertRaisesRegex(ValueError, "real-valued"):
            load_validated_source(sample, self.device)

    def test_float32_overflow_input(self):
        sample = self._make_sample()
        np.save(sample.face, np.full((10, 384), 1e300, dtype=np.float64))

        with self.assertRaisesRegex(ValueError, "float32 conversion"):
            load_validated_source(sample, self.device)

    def test_empty_array(self):
        sample = self._make_sample()
        np.save(sample.face, np.array([], dtype=np.float32).reshape(0, 384))

        with self.assertRaises(ValueError) as ctx:
            load_validated_source(sample, self.device)
        self.assertIn("empty", str(ctx.exception))


class TestHiddenStateExtraction(unittest.TestCase):
    def test_production_extraction_selects_hidden_state(self):
        from features.shubert_inference import extract_hidden_states

        length = 3
        source = [{"face": torch.zeros((length, 384))}]
        features = extract_hidden_states(FakeModel(), source, length)

        self.assertEqual(features.shape, (12, length, 768))
        np.testing.assert_array_equal(features[0], 0.0)
        np.testing.assert_array_equal(features[11], 11.0)


class TestMain(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.tmpdir = Path(tempfile.mkdtemp())
        self.sample = self._make_sample()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _make_sample(self, length=3):
        paths = [
            self.tmpdir / "sample_face.npy",
            self.tmpdir / "sample_hand1.npy",
            self.tmpdir / "sample_hand2.npy",
            self.tmpdir / "sample_pose.npy",
        ]
        for path in paths[:3]:
            np.save(path, np.zeros((length, 384), dtype=np.float32))
        np.save(paths[3], np.zeros((length, 14), dtype=np.float32))
        return SamplePaths("sample", *paths)

    def test_loads_each_sample_once(self):
        output_dir = self.tmpdir / "outputs"

        with patch("features.shubert_inference.load_model", return_value=FakeModel()), patch(
            "features.shubert_inference.load_validated_source",
            wraps=load_validated_source,
        ) as load_source:
            status = main([self.sample], Path("checkpoint.pt"), output_dir, None, self.device)

        self.assertEqual(status, 0)
        self.assertEqual(load_source.call_count, 1)

    def test_successful_run_clears_stale_failure_report(self):
        output_dir = self.tmpdir / "outputs"
        output_dir.mkdir()
        np.save(output_dir / "sample.npy", np.zeros((12, 3, 768), dtype=np.float32))
        failure_report = self.tmpdir / "failures.jsonl"
        failure_report.write_text('{"sample_id": "stale"}\n')

        with patch("features.shubert_inference.load_model", return_value=FakeModel()):
            status = main(
                [self.sample],
                Path("checkpoint.pt"),
                output_dir,
                failure_report,
                self.device,
            )

        self.assertEqual(status, 0)
        self.assertEqual(failure_report.read_text(), "")

    def test_checkpoint_failure_clears_stale_failure_report(self):
        failure_report = self.tmpdir / "failures.jsonl"
        failure_report.write_text('{"sample_id": "stale"}\n')

        with patch(
            "features.shubert_inference.load_model",
            side_effect=RuntimeError("bad checkpoint"),
        ), self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
            main(
                [self.sample],
                Path("checkpoint.pt"),
                self.tmpdir / "outputs",
                failure_report,
                self.device,
            )

        self.assertEqual(failure_report.read_text(), "")


class TestChunkSelection(unittest.TestCase):
    def test_negative_index_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            select_chunk([1, 2], index=-1, batch_size=1)

    def test_nonpositive_batch_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            select_chunk([1, 2], index=0, batch_size=0)


class TestCli(unittest.TestCase):
    def test_empty_manifest_clears_failure_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "empty.tsv"
            manifest.touch()
            failure_report = root / "failures.jsonl"
            failure_report.write_text('{"sample_id": "stale"}\n')
            project_root = Path(__file__).parents[1]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root / "fairseq")

            result = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "features" / "shubert_inference.py"),
                    "--csv_path",
                    str(manifest),
                    "--checkpoint_path",
                    str(root / "unused.pt"),
                    "--output_dir",
                    str(root / "outputs"),
                    "--failure_report",
                    str(failure_report),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(failure_report.read_text(), "")

class TestAtomicSave(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_valid_save(self):
        from features.shubert_inference import atomic_save
        output = self.tmpdir / "test.npy"
        features = np.random.randn(12, 10, 768).astype(np.float32)
        atomic_save(output, features, expected_t=10)
        self.assertTrue(output.exists())
        loaded = np.load(output)
        self.assertEqual(loaded.shape, (12, 10, 768))
        self.assertEqual(loaded.dtype, np.float32)

    def test_wrong_shape_rejected(self):
        from features.shubert_inference import atomic_save
        output = self.tmpdir / "test.npy"
        features = np.random.randn(12, 5, 768).astype(np.float32)
        with self.assertRaises(ValueError):
            atomic_save(output, features, expected_t=10)
        self.assertFalse(output.exists())

    def test_resume_valid(self):
        from features.shubert_inference import output_validation_error
        output = self.tmpdir / "test.npy"
        features = np.random.randn(12, 10, 768).astype(np.float32)
        np.save(output, features)
        error = output_validation_error(output, expected_t=10)
        self.assertIsNone(error)

    def test_resume_corrupt(self):
        from features.shubert_inference import output_validation_error
        output = self.tmpdir / "test.npy"
        features = np.random.randn(12, 5, 768).astype(np.float32)
        np.save(output, features)
        error = output_validation_error(output, expected_t=10)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("shape", error)

    def test_resume_zero_byte(self):
        from features.shubert_inference import output_validation_error
        output = self.tmpdir / "test.npy"
        output.touch()
        error = output_validation_error(output, expected_t=10)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("empty", error)


if __name__ == "__main__":
    unittest.main()
