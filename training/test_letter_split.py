"""Near-duplicate letter groups stay inside one split."""

import unittest

import numpy as np

from letter_split import group_indices, split_name


class LetterSplitTests(unittest.TestCase):
    def test_close_frames_share_a_group_and_a_split(self):
        base = np.zeros(4, dtype=np.float32)
        features = np.stack([base, base + 0.01, base + 2.0])
        groups = group_indices(features, threshold=0.35)
        self.assertEqual(len(groups), 2)
        together = next(group for group in groups if len(group) == 2)
        names = ["a.json", "b.json", "c.json"]
        key = min(names[index] for index in together)
        splits = {split_name(key) for _ in together}
        self.assertEqual(len(splits), 1)

    def test_split_name_is_stable(self):
        self.assertEqual(split_name("pose.json"), split_name("pose.json"))
        self.assertIn(split_name("pose.json"), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
