"""Augmentation keeps the feature width and mirror is an involution."""

import unittest

import numpy as np

from features import FEATURE_DIM, augment_sequence, mirror_sequence


class AugmentTests(unittest.TestCase):
    def test_mirror_negates_x_only(self):
        seq = np.array([[1, 2, 3, -4, 5, 6]], dtype=np.float32)
        flipped = mirror_sequence(seq)
        np.testing.assert_allclose(flipped, [[-1, 2, 3, 4, 5, 6]])
        np.testing.assert_allclose(mirror_sequence(flipped), seq)

    def test_augment_keeps_width(self):
        rng = np.random.default_rng(0)
        seq = rng.normal(size=(8, FEATURE_DIM)).astype(np.float32)
        out = augment_sequence(seq, rng)
        self.assertEqual(out.shape[1], FEATURE_DIM)
        self.assertGreaterEqual(out.shape[0], 3)

    def test_strong_augment_keeps_width(self):
        rng = np.random.default_rng(1)
        seq = rng.normal(size=(12, FEATURE_DIM)).astype(np.float32)
        for _ in range(8):
            out = augment_sequence(seq, rng, strength="strong")
            self.assertEqual(out.shape[1], FEATURE_DIM)
            self.assertGreaterEqual(out.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
