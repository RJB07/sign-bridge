"""Landmark sequence features shared by training and the browser.

A frame is wrist-centered, scaled by shoulder width (or palm size), and mirrored
so the dominant hand matches a right-hand frame. The browser port in
src/recognition/features.ts must stay in lockstep with this file.
"""

from __future__ import annotations

import numpy as np

FEATURE_DIM = 273
# Face-relative hands and pose, wrist-centered dominant shape, and velocities.
# 0:63 dominant vs nose, 63:126 other vs nose, 126:147 pose vs nose,
# 147:210 dominant shape vs wrist, 210:273 velocity of dominant vs nose,
# 273:294 velocity of pose vs nose. The browser port must match.
RICH_FEATURE_DIM = 294
# Rich vector, then wrist and five fingertips relative to lips, both eyes, and
# both cheek proxies (90), then five fingertip-to-palm cosines. The cheek
# proxy is the midpoint of the outer eye and the mouth corner in the pose.
FINE_OFFSET_DIM = 90
FINE_CURL_DIM = 5
FINE_XYZ = RICH_FEATURE_DIM + FINE_OFFSET_DIM
FINE_FEATURE_DIM = FINE_XYZ + FINE_CURL_DIM
FINE_HAND_INDEXES = (0, 4, 8, 12, 16, 20)
FINE_CURLS = ((1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
SEQ_LEN = 32
POSE_INDEXES = (0, 11, 12, 13, 14, 15, 16)
# Outer mouth, lips, and chin. Stored on the raw cache for ablations. The live
# feature vector does not include these until a run shows they raise test top-1.
LIP_INDEXES = (13, 14, 17, 61, 78, 81, 84, 87, 91, 146, 291, 308)


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
    arr = arr[:, :3]
    if not np.isfinite(arr).all():
        return None
    return arr


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


def _face_origin_and_scale(hand: np.ndarray | None, pose: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Nose-centered body frame. Shoulder width is the scale, palm size is the fallback."""
    scale = 0.0
    if pose is not None:
        scale = _xy_dist(pose[11], pose[12])
    if scale < 1e-4 and hand is not None:
        scale = _xy_dist(hand[0], hand[9])
    if scale < 1e-4:
        scale = 1.0
    if pose is not None:
        origin = pose[0].astype(np.float32).copy()
    elif hand is not None:
        origin = hand[0].astype(np.float32).copy()
    else:
        origin = np.zeros(3, dtype=np.float32)
    return origin, float(scale)


def sequence_features_rich(frames: list[dict], forced: str | None = None) -> np.ndarray:
    """Hand and arm coordinates relative to the nose, plus shape and velocity.

    The dominant hand is mirrored into a right-hand frame, same as
    `sequence_features`. Location is the hand relative to the nose (pose 0),
    scaled by shoulder width. Shape is the same hand relative to its wrist.
    The last 84 values are frame-to-frame velocity of the face-relative hand
    and the face-relative pose. Missing hands or a missing pose are zeros.
    Shape is (T, 294).
    """
    dominant = choose_dominant(frames, forced)
    if dominant is None:
        return np.zeros((0, RICH_FEATURE_DIM), dtype=np.float32)
    mirror = dominant == "Left"
    dom_key = "left" if dominant == "Left" else "right"
    other_key = "right" if dominant == "Left" else "left"
    rows = []
    prev_dom = None
    prev_pose = None
    for fr in frames:
        hand = _as_points(fr.get(dom_key), 21)
        other = _as_points(fr.get(other_key), 21)
        pose = _as_points(fr.get("pose"), 33)
        origin, scale = _face_origin_and_scale(hand, pose)
        dom_face = _pack(hand, origin, scale, mirror, 63)
        other_face = _pack(other, origin, scale, mirror, 63)
        if pose is None:
            pose_f = np.zeros(21, dtype=np.float32)
        else:
            pose_f = _pack(pose[list(POSE_INDEXES)], origin, scale, mirror, 21)
        if hand is None:
            shape = np.zeros(63, dtype=np.float32)
        else:
            shape = _pack(hand, hand[0], scale, mirror, 63)
        vel_dom = np.zeros(63, dtype=np.float32) if prev_dom is None else (dom_face - prev_dom)
        vel_pose = np.zeros(21, dtype=np.float32) if prev_pose is None else (pose_f - prev_pose)
        prev_dom = dom_face
        prev_pose = pose_f
        rows.append(np.concatenate([dom_face, other_face, pose_f, shape, vel_dom, vel_pose]).astype(np.float32))
    return np.stack(rows).astype(np.float32)


def _unit_dot(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    toward = b - a
    away = c - b
    left = float(np.linalg.norm(toward))
    right = float(np.linalg.norm(away))
    if left < 1e-6 or right < 1e-6:
        return 0.0
    return float(np.dot(toward, away) / (left * right))


def _face_targets(pose: np.ndarray | None, lips: np.ndarray | None, origin: np.ndarray, scale: float, mirror: bool) -> list[np.ndarray | None]:
    def placed(point: np.ndarray | None) -> np.ndarray | None:
        if point is None or not np.isfinite(point).all():
            return None
        rel = (point.astype(np.float32) - origin) / scale
        if mirror:
            rel = rel.copy()
            rel[0] *= -1
        return rel.astype(np.float32)

    lip_point = None
    if lips is not None and np.isfinite(lips).all():
        lip_point = lips.mean(axis=0)
    elif pose is not None:
        lip_point = (pose[9] + pose[10]) / 2
    eyes_cheeks: list[np.ndarray | None] = [lip_point]
    if pose is None:
        eyes_cheeks.extend([None, None, None, None])
    else:
        eyes_cheeks.append(pose[2])
        eyes_cheeks.append(pose[5])
        eyes_cheeks.append((pose[3] + pose[9]) / 2)
        eyes_cheeks.append((pose[6] + pose[10]) / 2)
    return [placed(point) for point in eyes_cheeks]


def sequence_features_fine(frames: list[dict], forced: str | None = None) -> np.ndarray:
    """Rich features plus hand-to-face offsets and fingertip curl.

    Offsets are the dominant wrist and five fingertips minus the lip center,
    each eye, and each cheek proxy. Curl is the cosine at each finger joint.
    Shape is (T, 389). The first 294 values match `sequence_features_rich`.
    """
    rich = sequence_features_rich(frames, forced)
    if rich.shape[0] == 0:
        return np.zeros((0, FINE_FEATURE_DIM), dtype=np.float32)
    dominant = choose_dominant(frames, forced)
    mirror = dominant == "Left"
    dom_key = "left" if dominant == "Left" else "right"
    extras = []
    for fr in frames:
        hand = _as_points(fr.get(dom_key), 21)
        pose = _as_points(fr.get("pose"), 33)
        lips = _as_points(fr.get("lips"), len(LIP_INDEXES))
        origin, scale = _face_origin_and_scale(hand, pose)
        targets = _face_targets(pose, lips, origin, scale, mirror)
        offsets = []
        if hand is None:
            offsets.append(np.zeros(FINE_OFFSET_DIM, dtype=np.float32))
            curls = np.zeros(FINE_CURL_DIM, dtype=np.float32)
        else:
            placed_hand = []
            for index in FINE_HAND_INDEXES:
                rel = (hand[index] - origin) / scale
                if mirror:
                    rel = rel.copy()
                    rel[0] *= -1
                placed_hand.append(rel.astype(np.float32))
            for point in placed_hand:
                for target in targets:
                    if target is None:
                        offsets.append(np.zeros(3, dtype=np.float32))
                    else:
                        offsets.append((point - target).astype(np.float32))
            curls = np.array([_unit_dot(hand[a], hand[b], hand[c]) for a, b, c in FINE_CURLS], dtype=np.float32)
        extras.append(np.concatenate([*offsets, curls]).astype(np.float32))
    return np.concatenate([rich, np.stack(extras)], axis=1).astype(np.float32)


def frames_from_raw(raw: np.ndarray) -> list[dict]:
    """Inverse of the cache pack: left, right, pose, then the lip subset."""
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"raw landmarks must be (T, points, 3), got {array.shape}")
    frames = []
    layout = (("left", 0, 21), ("right", 21, 21), ("pose", 42, 33), ("lips", 75, len(LIP_INDEXES)))
    for time in range(array.shape[0]):
        frame: dict = {}
        for key, start, count in layout:
            if start + count > array.shape[1]:
                frame[key] = None
                continue
            block = array[time, start : start + count, :3]
            frame[key] = None if block.shape != (count, 3) or not np.isfinite(block).all() else block.astype(np.float32)
        frames.append(frame)
    return frames


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


def mirror_sequence(seq: np.ndarray, xyz_dims: int | None = None) -> np.ndarray:
    """Flip the horizontal axis of an already-normalized feature sequence.

    `xyz_dims` limits the flip to a coordinate prefix. Scalar tail channels,
    such as fingertip curl, stay put.
    """
    out = np.array(seq, dtype=np.float32, copy=True)
    width = out.shape[-1] if xyz_dims is None else min(xyz_dims, out.shape[-1])
    out[..., 0:width:3] *= -1
    return out


def _rotate_scale(seq: np.ndarray, angle: float, scale: float, xyz_dims: int | None = None) -> np.ndarray:
    width = seq.shape[-1] if xyz_dims is None else xyz_dims
    cosine, sine = np.cos(angle), np.sin(angle)
    head = np.array(seq[:, :width], dtype=np.float32, copy=True)
    flat = head.reshape(seq.shape[0], -1, 3)
    x = flat[:, :, 0].copy()
    y = flat[:, :, 1].copy()
    flat[:, :, 0] = cosine * x - sine * y
    flat[:, :, 1] = sine * x + cosine * y
    head = (flat.reshape(seq.shape[0], -1) * scale).astype(np.float32)
    if width == seq.shape[-1]:
        return head
    return np.concatenate([head, seq[:, width:]], axis=1).astype(np.float32)


def _shift_positions(seq: np.ndarray, position_dims: int, shift: np.ndarray) -> np.ndarray:
    out = np.array(seq, dtype=np.float32, copy=True)
    flat = out[:, :position_dims].reshape(seq.shape[0], -1, 3)
    flat += shift
    out[:, :position_dims] = flat.reshape(seq.shape[0], position_dims)
    return out


def _augment_rich(seq: np.ndarray, rng: np.random.Generator, layout: str) -> np.ndarray:
    """Wider rotation, scale, frame drop, and whole-hand dropout.

    `layout="rich"` is the 294-d face-relative vector. `layout="legacy"` is the
    273-d vector. Position channels are shifted; velocity channels are not.
    """
    if layout == "fine":
        expected = FINE_FEATURE_DIM
        xyz_dims = FINE_XYZ
    elif layout == "rich":
        expected = RICH_FEATURE_DIM
        xyz_dims = None
    elif layout == "legacy":
        expected = FEATURE_DIM
        xyz_dims = None
    else:
        raise ValueError(f"unknown feature layout {layout}")
    seq = np.array(seq, dtype=np.float32, copy=True)
    if seq.shape[1] != expected:
        raise ValueError(f"{layout} augmentation expected width {expected}, got {seq.shape[1]}")
    if seq.shape[0] > 4 and rng.random() < 0.9:
        low = max(3, int(seq.shape[0] * 0.5))
        high = max(low + 1, int(round(seq.shape[0] * 1.7)) + 1)
        high = min(48, high)
        low = min(low, high - 1)
        seq = resample_sequence(seq, int(rng.integers(low, high)))
    if seq.shape[0] > 6 and rng.random() < 0.6:
        span = int(rng.integers(max(4, int(seq.shape[0] * 0.6)), seq.shape[0] + 1))
        start = int(rng.integers(0, seq.shape[0] - span + 1))
        seq = seq[start : start + span]
    if seq.shape[0] > 7 and rng.random() < 0.7:
        n_drop = 1 if rng.random() < 0.55 else min(2, seq.shape[0] - 4)
        if n_drop > 0:
            drop_at = rng.choice(seq.shape[0], size=n_drop, replace=False)
            seq = np.delete(seq, drop_at, axis=0)
    if rng.random() < 0.5:
        seq = mirror_sequence(seq, xyz_dims)
    seq = _rotate_scale(seq, float(rng.uniform(-0.55, 0.55)), float(rng.uniform(0.65, 1.4)), xyz_dims)
    if rng.random() < 0.8:
        shift = rng.uniform(-0.12, 0.12, size=(1, 1, 3)).astype(np.float32)
        position_dims = 210 if layout == "rich" else 147
        seq = _shift_positions(seq, position_dims, shift)
    if rng.random() < 0.55:
        seq[:, 63:126] = 0
        if layout == "legacy":
            seq[:, 210:273] = 0
    drops = int(rng.integers(0, 4))
    bases = (0, 147, 210) if layout == "rich" else (0, 147)
    for _ in range(drops):
        landmark = int(rng.integers(0, 21))
        for base in bases:
            seq[:, base + landmark * 3 : base + landmark * 3 + 3] = 0
    seq += rng.normal(0.0, 0.02, size=seq.shape).astype(np.float32)
    return seq.astype(np.float32)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator, strength: str = "base", layout: str = "legacy") -> np.ndarray:
    """Time warp, mirror, rotation, scale, dropped landmarks, and light noise.

    `strength="rich"` adds wider rotation and scale, extra frame drops, and
    whole-hand dropout. `layout` selects which channels are position, velocity,
    and the other hand. `strength="strong"` keeps the previous wider ranges.

    `strength="strong"` widens those ranges, crops time, shifts the position
    channels (not the velocity block), and drops more landmarks. PopSign clips
    are short, so a crop never goes below four frames. Mirror stands in for a
    left-hand signer.
    """
    if layout == "fine" or strength == "rich":
        return _augment_rich(seq, rng, "fine" if layout == "fine" else layout)
    seq = np.array(seq, dtype=np.float32, copy=True)
    strong = strength == "strong"
    if seq.shape[0] > 4 and rng.random() < (0.85 if strong else 0.75):
        low_frac = 0.55 if strong else 0.75
        high_frac = 1.6 if strong else 1.35
        low = max(3, int(seq.shape[0] * low_frac))
        high = max(low + 1, int(round(seq.shape[0] * high_frac)) + 1)
        # Cap the warp at 48 frames. A long non-game clip can make `low` itself
        # land on that cap, and NumPy rejects integers(low, high) when low >= high.
        high = min(48, high)
        low = min(low, high - 1)
        seq = resample_sequence(seq, int(rng.integers(low, high)))
    if strong and seq.shape[0] > 6 and rng.random() < 0.55:
        span = int(rng.integers(max(4, int(seq.shape[0] * 0.65)), seq.shape[0] + 1))
        start = int(rng.integers(0, seq.shape[0] - span + 1))
        seq = seq[start : start + span]
    if seq.shape[0] > 6 and rng.random() < (0.5 if strong else 0.35):
        seq = np.delete(seq, int(rng.integers(0, seq.shape[0])), axis=0)
    if rng.random() < 0.5:
        seq = mirror_sequence(seq)
    angle = float(rng.uniform(-0.4, 0.4) if strong else rng.uniform(-0.28, 0.28))
    cosine, sine = np.cos(angle), np.sin(angle)
    flat = seq.reshape(seq.shape[0], -1, 3)
    x = flat[:, :, 0].copy()
    y = flat[:, :, 1].copy()
    flat[:, :, 0] = cosine * x - sine * y
    flat[:, :, 1] = sine * x + cosine * y
    seq = flat.reshape(seq.shape[0], -1)
    seq *= float(rng.uniform(0.75, 1.25) if strong else rng.uniform(0.85, 1.15))
    if strong and seq.shape[1] >= 147 and (seq.shape[1] - 126) % 3 == 0:
        n_pos = (seq.shape[1] - 126) // 3
        shift = rng.uniform(-0.1, 0.1, size=(1, 1, 3)).astype(np.float32)
        flat = seq.reshape(seq.shape[0], -1, 3)
        flat[:, :n_pos, :] += shift
        seq = flat.reshape(seq.shape[0], -1)
    drops = 2 if strong and rng.random() < 0.7 else (1 if rng.random() < 0.3 else 0)
    for _ in range(drops):
        landmark = int(rng.integers(0, 21))
        seq[:, landmark * 3 : landmark * 3 + 3] = 0
        if seq.shape[1] >= 210:
            seq[:, 147 + landmark * 3 : 147 + landmark * 3 + 3] = 0
    noise = 0.02 if strong else 0.012
    seq += rng.normal(0.0, noise, size=seq.shape).astype(np.float32)
    return seq.astype(np.float32)
