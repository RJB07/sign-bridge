"""Landmark sequence features shared by training and the browser.

A frame is wrist-centered, scaled by shoulder width (or palm size), and mirrored
so the dominant hand matches a right-hand frame. The browser port in
src/recognition/features.ts must stay in lockstep with this file.
"""

from __future__ import annotations

import numpy as np

FEATURE_DIM = 273
SEQ_LEN = 32
POSE_INDEXES = (0, 11, 12, 13, 14, 15, 16)


def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _as_points(value, expected: int | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    if expected is not None and arr.shape[0] < expected:
        return None
    return arr[:, :3]


def choose_dominant(frames: list[dict], forced: str | None = None) -> str | None:
    if forced in ("Left", "Right"):
        key = "left" if forced == "Left" else "right"
        if any(_as_points(fr.get(key), 21) is not None for fr in frames):
            return forced

    motion = {"Left": 0.0, "Right": 0.0}
    count = {"Left": 0, "Right": 0}
    prev = {"Left": None, "Right": None}
    for fr in frames:
        for side, key in (("Left", "left"), ("Right", "right")):
            hand = _as_points(fr.get(key), 21)
            if hand is None:
                continue
            count[side] += 1
            if prev[side] is not None:
                motion[side] += _xy_dist(hand[0], prev[side][0])
            prev[side] = hand
    if count["Left"] == 0 and count["Right"] == 0:
        return None
    if motion["Left"] > motion["Right"] + 1e-6:
        return "Left"
    if motion["Right"] > motion["Left"] + 1e-6:
        return "Right"
    return "Left" if count["Left"] >= count["Right"] else "Right"


def _origin_and_scale(hand: np.ndarray | None, pose: np.ndarray | None) -> tuple[np.ndarray, float]:
    if hand is not None:
        origin = hand[0].astype(np.float32).copy()
    elif pose is not None:
        origin = ((pose[11] + pose[12]) / 2).astype(np.float32)
    else:
        origin = np.zeros(3, dtype=np.float32)

    scale = 0.0
    if pose is not None:
        scale = _xy_dist(pose[11], pose[12])
    if scale < 1e-4 and hand is not None:
        scale = _xy_dist(hand[0], hand[9])
    if scale < 1e-4:
        scale = 1.0
    return origin, float(scale)


def _normalize(points: np.ndarray, origin: np.ndarray, scale: float, mirror: bool) -> np.ndarray:
    rel = (points - origin) / scale
    if mirror:
        rel = rel.copy()
        rel[:, 0] *= -1
    return rel.astype(np.float32)


def _pack(points: np.ndarray | None, origin: np.ndarray, scale: float, mirror: bool, width: int) -> np.ndarray:
    if points is None:
        return np.zeros(width, dtype=np.float32)
    return _normalize(points, origin, scale, mirror).reshape(-1).astype(np.float32)


def sequence_features(frames: list[dict], forced: str | None = None) -> np.ndarray:
    """Both hands plus arm pose, mirrored so the moving hand is on the right.

    Layout per frame: dominant hand, other hand, 7 pose points, dominant delta,
    other-hand delta. Missing hands are zeros. Shape is (T, 273).
    """
    dominant = choose_dominant(frames, forced)
    if dominant is None:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    mirror = dominant == "Left"
    dom_key = "left" if dominant == "Left" else "right"
    other_key = "right" if dominant == "Left" else "left"
    rows = []
    prev_dom = None
    prev_other = None
    for fr in frames:
        hand = _as_points(fr.get(dom_key), 21)
        other = _as_points(fr.get(other_key), 21)
        pose = _as_points(fr.get("pose"), 33)
        origin, scale = _origin_and_scale(hand, pose)
        dom = _pack(hand, origin, scale, mirror, 63)
        nondom = _pack(other, origin, scale, mirror, 63)
        delta_dom = np.zeros(63, dtype=np.float32) if prev_dom is None else (dom - prev_dom)
        delta_other = np.zeros(63, dtype=np.float32) if prev_other is None else (nondom - prev_other)
        prev_dom = dom
        prev_other = nondom
        if pose is None:
            pose_f = np.zeros(21, dtype=np.float32)
        else:
            pose_f = _pack(pose[list(POSE_INDEXES)], origin, scale, mirror, 21)
        rows.append(np.concatenate([dom, nondom, pose_f, delta_dom, delta_other]).astype(np.float32))
    return np.stack(rows).astype(np.float32)


def resample_sequence(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[0] == 0:
        return np.zeros((0, FEATURE_DIM if seq.ndim != 2 else seq.shape[1]), dtype=np.float32)
    if seq.shape[0] == length:
        return seq.astype(np.float32)
    if seq.shape[0] == 1:
        return np.repeat(seq, length, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, seq.shape[0])
    dst = np.linspace(0.0, 1.0, length)
    out = np.empty((length, seq.shape[1]), dtype=np.float32)
    for col in range(seq.shape[1]):
        out[:, col] = np.interp(dst, src, seq[:, col])
    return out


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Flip the horizontal axis of an already-normalized feature sequence."""
    out = np.array(seq, dtype=np.float32, copy=True)
    out[..., 0::3] *= -1
    return out


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time warp, mirror, rotation, scale, one dropped landmark, and light noise.

    PopSign clips are only a handful of frames, so the ranges stay tight enough
    that the gloss is still recognizable. Mirror is the left-hand stand-in.
    """
    seq = np.array(seq, dtype=np.float32, copy=True)
    if seq.shape[0] > 4 and rng.random() < 0.75:
        low = max(3, int(seq.shape[0] * 0.75))
        high = min(48, max(low + 1, int(round(seq.shape[0] * 1.35)) + 1))
        seq = resample_sequence(seq, int(rng.integers(low, high)))
    if seq.shape[0] > 6 and rng.random() < 0.35:
        drop = int(rng.integers(0, seq.shape[0]))
        seq = np.delete(seq, drop, axis=0)
    if rng.random() < 0.5:
        seq = mirror_sequence(seq)
    angle = float(rng.uniform(-0.28, 0.28))
    c, s = np.cos(angle), np.sin(angle)
    flat = seq.reshape(seq.shape[0], -1, 3)
    x = flat[:, :, 0].copy()
    y = flat[:, :, 1].copy()
    flat[:, :, 0] = c * x - s * y
    flat[:, :, 1] = s * x + c * y
    seq = flat.reshape(seq.shape[0], -1)
    seq *= float(rng.uniform(0.85, 1.15))
    if rng.random() < 0.3:
        landmark = int(rng.integers(0, 21))
        seq[:, landmark * 3 : landmark * 3 + 3] = 0
        if seq.shape[1] >= 210:
            seq[:, 147 + landmark * 3 : 147 + landmark * 3 + 3] = 0
    seq += rng.normal(0.0, 0.012, size=seq.shape).astype(np.float32)
    return seq.astype(np.float32)
