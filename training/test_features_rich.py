"""Face-relative features stay finite and line up with the raw landmark pack."""

import unittest

import numpy as np

from features import RICH_FEATURE_DIM, frames_from_raw, sequence_features_rich


def _hand(x: float, y: float) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = x
    points[:, 1] = y
    points[:, 2] = 0.01
    for index in range(21):
        points[index, 0] += index * 0.002
        points[index, 1] += index * 0.001
    return points


def _pose(nose_x: float, nose_y: float) -> np.ndarray:
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[:, 0] = 0.5
    pose[:, 1] = 0.4
    pose[0, 0] = nose_x
    pose[0, 1] = nose_y
    pose[11, 0] = 0.4
    pose[12, 0] = 0.6
    pose[11, 1] = pose[12, 1] = 0.45
    return pose


def _frames(hand_x: float) -> list[dict]:
    frames = []
    for step in range(3):
        frames.append(
            {
                "left": None,
                "right": _hand(hand_x, 0.3 + step * 0.05),
                "pose": _pose(0.5, 0.2),
            }
        )
    return frames


class RichFeatureTests(unittest.TestCase):
    def test_width_velocity_and_face_offset(self):
        features = sequence_features_rich(_frames(0.7))
        self.assertEqual(features.shape, (3, RICH_FEATURE_DIM))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.allclose(features[0, 210:], 0))
        self.assertGreater(abs(float(features[0, 0])), 0.2)
        self.assertFalse(np.allclose(features[1, 210:273], 0))
        self.assertAlmostEqual(float(features[0, 147]), 0.0, places=5)

    def test_left_hand_is_mirrored_to_the_right(self):
        source = _frames(0.7)
        right = sequence_features_rich(source)
        left_frames = []
        for frame in source:
            flipped = {
                "left": frame["right"].copy(),
                "right": None,
                "pose": frame["pose"].copy(),
            }
            flipped["left"][:, 0] = 1.0 - frame["right"][:, 0]
            flipped["pose"][:, 0] = 1.0 - frame["pose"][:, 0]
            left_frames.append(flipped)
        left = sequence_features_rich(left_frames)
        np.testing.assert_allclose(left, right, atol=1e-5)

    def test_raw_pack_roundtrip(self):
        raw = np.full((2, 87, 3), np.nan, dtype=np.float32)
        raw[:, 21:42] = _hand(0.66, 0.4)
        raw[:, 42:75] = _pose(0.5, 0.2)
        frames = frames_from_raw(raw)
        self.assertIsNone(frames[0]["left"])
        self.assertIsNone(frames[0]["lips"])
        features = sequence_features_rich(frames)
        self.assertEqual(features.shape[1], RICH_FEATURE_DIM)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
