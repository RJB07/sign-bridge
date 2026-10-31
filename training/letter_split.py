"""Near-duplicate groups for the static letter set.

sid220/asl-now-fingerspelling has no participant id. Frames of the same pose
are often a fraction of a unit apart, so a filename hash puts copies in both
train and test. Groups keep those copies together. This is not a verified
signer split.
"""

from __future__ import annotations

import hashlib

import numpy as np

NEAR_DUPLICATE = 0.35


def split_name(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def group_indices(features: np.ndarray, threshold: float = NEAR_DUPLICATE) -> list[list[int]]:
    count = len(features)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(count):
        for right in range(left + 1, count):
            if float(np.linalg.norm(features[left] - features[right])) <= threshold:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parent[right_root] = left_root
    buckets: dict[int, list[int]] = {}
    for index in range(count):
        buckets.setdefault(find(index), []).append(index)
    return list(buckets.values())
