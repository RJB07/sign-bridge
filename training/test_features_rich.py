"""Face-relative features stay finite and line up with the raw landmark pack."""

import unittest

import numpy as np

from features import (
    FINE_FEATURE_DIM,
    RICH_FEATURE_DIM,
    frames_from_raw,
    sequence_features_fine,
    sequence_features_rich,
    sequence_features_signer,
)


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

    def test_fine_features_extend_the_rich_vector(self):
        frames = _frames(0.7)
        frames[0]["lips"] = np.stack([_pose(0.5, 0.2)[0] for _ in range(12)])
        rich = sequence_features_rich(frames)
        fine = sequence_features_fine(frames)
        self.assertEqual(fine.shape, (3, FINE_FEATURE_DIM))
        np.testing.assert_allclose(fine[:, :RICH_FEATURE_DIM], rich)
        self.assertTrue(np.isfinite(fine).all())
        curls = fine[:, -5:]
        self.assertTrue(np.all(np.abs(curls) <= 1))

    def test_palm_shape_ignores_hand_size(self):
        base = _frames(0.7)
        grown = []
        for frame in base:
            hand = frame["right"].copy()
            wrist = hand[0].copy()
            hand = wrist + 1.8 * (hand - wrist)
            grown.append({"left": None, "right": hand, "pose": frame["pose"].copy()})
        signer_base = sequence_features_signer(base)
        signer_grown = sequence_features_signer(grown)
        np.testing.assert_allclose(signer_base[:, 147:210], signer_grown[:, 147:210], atol=1e-4)
        fine_base = sequence_features_fine(base)
        fine_grown = sequence_features_fine(grown)
        self.assertFalse(np.allclose(fine_base[:, 147:210], fine_grown[:, 147:210], atol=1e-3))
        self.assertEqual(signer_base.shape[1], FINE_FEATURE_DIM)

    def test_left_hand_signer_matches_the_right_hand(self):
        source = _frames(0.7)
        right = sequence_features_signer(source)
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
        left = sequence_features_signer(left_frames)
        np.testing.assert_allclose(left, right, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
