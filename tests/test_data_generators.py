"""CSV and normal generators separate saved instances from evolving sampling state."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments_wo_stress.data import CSVDataGenerator, NormalDataGenerator


class DataGeneratorTests(unittest.TestCase):
    """Describe input data once, then restore generation state without rereading inputs."""

    def test_csv_instance_retains_source_data_and_digest_without_rereading(self):
        """The saved CSV instance remains sufficient after the original file is removed."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            payload = b"x,y\n1,2\n3,4\n"
            path.write_bytes(payload)
            instance = CSVDataGenerator.create_instance(
                rng=np.random.default_rng(0), path=path, skip_header=1
            )
            self.assertEqual(instance.kind, "dataset")
            self.assertEqual(instance.metadata["sha256"], hashlib.sha256(payload).hexdigest())
            path.unlink()
            source = CSVDataGenerator(rng=np.random.default_rng(0), instance=instance)
            np.testing.assert_array_equal(source.generate(None), [[1, 2], [3, 4]])
            self.assertIs(source.values, instance.arrays["data"])

    def test_normal_instance_describes_distribution_and_keeps_sampling_separate(self):
        """Describing a normal distribution consumes no instance randomness."""
        instance_rng = np.random.default_rng(0)
        before = copy.deepcopy(instance_rng.bit_generator.state)
        instance = NormalDataGenerator.create_instance(
            rng=instance_rng, size=[2, 3], loc=4, scale=0
        )
        self.assertEqual(instance_rng.bit_generator.state, before)
        self.assertEqual(instance.kind, "normal_distribution")
        self.assertEqual(instance.metadata["size"], (2, 3))
        generator = NormalDataGenerator(rng=np.random.default_rng(1), instance=instance)
        np.testing.assert_array_equal(generator.generate(None), np.full((2, 3), 4))

    def test_csv_batches_resume_without_storing_dataset(self):
        """CSV checkpoints save only a valid cursor and resume at the next batch."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            np.savetxt(path, np.arange(12).reshape(6, 2), delimiter=",")
            source = CSVDataGenerator(rng=np.random.default_rng(0), path=path)
            np.testing.assert_array_equal(source.generate(2), [[0, 1], [2, 3]])
            restored = CSVDataGenerator(rng=np.random.default_rng(0), path=path)
            restored.load_state_dict(source.state_dict())
            np.testing.assert_array_equal(restored.generate(2), [[4, 5], [6, 7]])
            self.assertFalse(source.generate(None).flags.writeable)
            self.assertEqual(source.state_dict(), {"cursor": 2})
            with self.assertRaises(ValueError):
                restored.load_state_dict({"cursor": 10})
