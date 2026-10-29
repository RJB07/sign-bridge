"""ASL Signs conversion and signer split. No Kaggle download."""

import unittest

import numpy as np

from asl_signs import credential_env, frames_from_records, signer_split
from features import FEATURE_DIM, sequence_features
from splits import leakage_report


def _row(frame: int, kind: str, index: int, x: float, y: float, z: float = 0.0) -> dict:
    return {"frame": frame, "type": kind, "landmark_index": index, "x": x, "y": y, "z": z}


def _sequence() -> list[dict]:
    records = []
    for frame in range(4):
        for index in range(21):
            records.append(_row(frame, "right_hand", index, 0.6 + frame * 0.01, 0.4 + index * 0.001, 0.01))
            records.append(_row(frame, "left_hand", index, 0.3, 0.4, 0.01))
        for index in range(33):
            records.append(_row(frame, "pose", index, 0.5 + (0.1 if index == 12 else 0.0), 0.3, 0.0))
        records.append(_row(frame, "face", 0, 0.5, 0.2, 0.0))
    return records


class AslSignsTests(unittest.TestCase):
    def test_frames_become_the_shared_feature_width(self):
        frames = frames_from_records(_sequence())
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[0]["right"].shape, (21, 3))
        self.assertIsNone(frames[0].get("face"))
        features = sequence_features(frames)
        self.assertEqual(features.shape[1], FEATURE_DIM)
        self.assertGreater(features.shape[0], 0)
        self.assertTrue(np.isfinite(features).all())

    def test_missing_hand_is_dropped(self):
        records = _sequence()
        records = [row for row in records if row["type"] != "left_hand"]
        for row in records:
            if row["type"] == "right_hand" and row["landmark_index"] == 0:
                row["x"] = float("nan")
        frames = frames_from_records(records)
        self.assertIsNone(frames[0]["right"])
        self.assertIsNone(frames[0]["left"])

    def test_signer_split_has_no_overlap(self):
        mapping = signer_split([str(i) for i in range(21)])
        rows = [{"split": split, "signer": signer, "label": "hello"} for signer, split in mapping.items()]
        report = leakage_report(rows)
        self.assertEqual(report["train&validation"], [])
        self.assertEqual(report["train&test"], [])
        self.assertEqual(report["validation&test"], [])
        self.assertIn("train", set(mapping.values()))
        self.assertIn("validation", set(mapping.values()))
        self.assertIn("test", set(mapping.values()))

    def test_missing_token_is_absent(self):
        import os

        saved = {name: os.environ.pop(name, None) for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")}
        try:
            self.assertIsNone(credential_env())
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
