"""Signer-split invariants that do not need the downloaded videos."""

import unittest

from splits import (
    assert_no_signer_leakage,
    labels_in_every_split,
    leakage_report,
    select_balanced,
    signer_from_path,
)


class SplitTests(unittest.TestCase):
    def test_signer_parse(self):
        path = "train/TV/4a.6001-TV-2022_12_14_14_59_01.480-0.mp4"
        self.assertEqual(signer_from_path(path, "TV"), "4a.6001")

    def test_leakage_detected(self):
        rows = [
            {"split": "train", "signer": "a", "label": "apple"},
            {"split": "test", "signer": "a", "label": "apple"},
        ]
        self.assertEqual(leakage_report(rows)["train&test"], ["a"])
        with self.assertRaises(SystemExit):
            assert_no_signer_leakage(rows)

    def test_balanced_selection_keeps_signers(self):
        rows = []
        for signer in ("s1", "s2", "s3"):
            for i in range(10):
                rows.append({"split": "train", "label": "apple", "signer": signer, "i": i})
        picked = select_balanced(rows, ["apple"], {"train": 6}, seed=1)
        signers = {row["signer"] for row in picked}
        self.assertEqual(len(picked), 6)
        self.assertEqual(signers, {"s1", "s2", "s3"})

    def test_label_must_exist_in_every_split(self):
        rows = [
            {"split": "train", "label": "apple"},
            {"split": "validation", "label": "apple"},
            {"split": "test", "label": "bed"},
            {"split": "train", "label": "bed"},
            {"split": "validation", "label": "bed"},
            {"split": "test", "label": "bed"},
        ]
        self.assertEqual(labels_in_every_split(rows), ["bed"])


if __name__ == "__main__":
    unittest.main()
