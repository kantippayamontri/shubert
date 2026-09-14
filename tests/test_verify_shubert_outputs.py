import tempfile
from pathlib import Path
import unittest

import numpy as np

from features.verify_shubert_outputs import ManifestSample, select_chunk, verify_outputs


class TestVerifyOutputs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.output_dir = self.tmpdir / "outputs"
        self.output_dir.mkdir()
        self.sample = self._make_sample()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _make_sample(self, length=3, sample_id="sample"):
        paths = [
            self.tmpdir / f"{sample_id}_face.npy",
            self.tmpdir / f"{sample_id}_hand1.npy",
            self.tmpdir / f"{sample_id}_hand2.npy",
            self.tmpdir / f"{sample_id}_pose.npy",
        ]
        for path in paths[:3]:
            np.save(path, np.zeros((length, 384), dtype=np.float32))
        np.save(paths[3], np.zeros((length, 14), dtype=np.float32))
        return ManifestSample(sample_id, *paths)

    def _write_valid_output(self):
        np.save(self.output_dir / "sample.npy", np.zeros((12, 3, 768), dtype=np.float32))

    def test_exact_corpus_passes(self):
        self._write_valid_output()

        result = verify_outputs(self.output_dir, [self.sample])

        self.assertTrue(result.ok)
        self.assertEqual(result.valid, 1)

    def test_unexpected_output_fails(self):
        self._write_valid_output()
        np.save(self.output_dir / "extra.npy", np.zeros((12, 3, 768), dtype=np.float32))

        result = verify_outputs(self.output_dir, [self.sample])

        self.assertFalse(result.ok)
        self.assertEqual(result.unexpected, ["extra.npy"])

    def test_duplicate_sample_id_fails(self):
        self._write_valid_output()

        result = verify_outputs(self.output_dir, [self.sample, self.sample])

        self.assertFalse(result.ok)
        self.assertEqual(result.duplicates, ["sample"])

    def test_mismatched_input_stream_fails(self):
        self._write_valid_output()
        np.save(self.sample.body, np.zeros((2, 14), dtype=np.float32))

        result = verify_outputs(self.output_dir, [self.sample])

        self.assertFalse(result.ok)
        self.assertEqual(result.invalid, 1)

    def test_complex_input_fails(self):
        self._write_valid_output()
        np.save(self.sample.face, np.ones((3, 384), dtype=np.complex64))

        result = verify_outputs(self.output_dir, [self.sample])

        self.assertFalse(result.ok)
        self.assertEqual(result.invalid, 1)

    def test_chunk_allows_other_manifest_outputs(self):
        self._write_valid_output()
        self._make_sample(sample_id="other")
        np.save(self.output_dir / "other.npy", np.zeros((12, 3, 768), dtype=np.float32))

        result = verify_outputs(
            self.output_dir,
            [self.sample],
            allowed_output_names={"sample.npy", "other.npy"},
        )

        self.assertTrue(result.ok)

    def test_invalid_chunk_bounds_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            select_chunk([self.sample], index=-1, batch_size=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            select_chunk([self.sample], index=0, batch_size=0)


if __name__ == "__main__":
    unittest.main()
