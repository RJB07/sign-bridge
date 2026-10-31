"""Google ASL Signs (Kaggle competition `asl-signs`) → SignBridge cache.

The download needs a token. Either set `KAGGLE_API_TOKEN`, or set both
`KAGGLE_USERNAME` and `KAGGLE_KEY`. Nothing is downloaded unless one of those
is present. Landmark rows use the competition schema: frame, type
(`left_hand`, `right_hand`, `pose`, `face`), landmark_index, x, y, z.
"""

from __future__ import annotations

import csv
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from features import sequence_features

HAND = 21
POSE = 33


def credential_env() -> dict[str, str] | None:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return {"KAGGLE_API_TOKEN": token}
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        return {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key}
    return None


def signer_split(participant_ids: list[str]) -> dict[str, str]:
    """Deterministic signer-disjoint split. Entire participants stay together."""
    ids = sorted({str(participant) for participant in participant_ids})
    count = len(ids)
    if count < 3:
        raise ValueError("Need at least 3 participants for a signer-disjoint split")
    n_test = max(1, round(count * 0.2))
    n_val = max(1, round(count * 0.15))
    while n_test + n_val >= count:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    test_ids = set(ids[-n_test:])
    val_ids = set(ids[-(n_test + n_val) : -n_test])
    mapping: dict[str, str] = {}
    for participant in ids:
        if participant in test_ids:
            mapping[participant] = "test"
        elif participant in val_ids:
            mapping[participant] = "validation"
        else:
            mapping[participant] = "train"
    return mapping


def _clean(points: np.ndarray) -> np.ndarray | None:
    if points.size == 0 or not np.isfinite(points).all():
        return None
    return points.astype(np.float32)


def frames_from_records(records: list[dict]) -> list[dict]:
    """Long-form landmark rows for one sequence, in competition column names."""
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["frame"]), []).append(record)
    frames = []
    for frame_id in sorted(grouped):
        left = np.full((HAND, 3), np.nan, dtype=np.float32)
        right = np.full((HAND, 3), np.nan, dtype=np.float32)
        pose = np.full((POSE, 3), np.nan, dtype=np.float32)
        targets = {"left_hand": left, "right_hand": right, "pose": pose}
        for record in grouped[frame_id]:
            target = targets.get(str(record["type"]))
            if target is None:
                continue
            index = int(record["landmark_index"])
            if index < 0 or index >= target.shape[0]:
                continue
            target[index] = (float(record["x"]), float(record["y"]), float(record["z"]))
        frames.append({"left": _clean(left), "right": _clean(right), "pose": _clean(pose)})
    return frames


def read_parquet_records(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    columns = table.to_pydict()
    count = len(columns["frame"])
    return [
        {
            "frame": columns["frame"][i],
            "type": columns["type"][i],
            "landmark_index": columns["landmark_index"][i],
            "x": columns["x"][i],
            "y": columns["y"][i],
            "z": columns["z"][i],
        }
        for i in range(count)
    ]


def download_competition(dest: Path) -> None:
    creds = credential_env()
    if creds is None:
        raise SystemExit(
            "Set KAGGLE_API_TOKEN, or both KAGGLE_USERNAME and KAGGLE_KEY, "
            "before downloading the asl-signs competition."
        )
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "train.csv").exists():
        return
    env = os.environ.copy()
    env.update(creds)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "asl-signs", "-p", str(dest)],
        check=True,
        env=env,
    )
    for archive in dest.glob("*.zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(dest)


def build_cache(source: Path, cache: Path, limit: int | None = None) -> dict[str, int]:
    """Write index.jsonl and seq/*.npy in the format train_sequence_model.py reads."""
    import json

    from splits import assert_no_signer_leakage

    csv_path = source / "train.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run the download first.")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    mapping = signer_split([row["participant_id"] for row in rows])
    seq_dir = cache / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache / "index.jsonl"
    counts = {"train": 0, "validation": 0, "test": 0, "skipped": 0}
    written = []
    with index_path.open("w", encoding="utf-8") as handle:
        for number, row in enumerate(rows):
            participant = str(row["participant_id"])
            split = mapping[participant]
            parquet_path = source / row["path"]
            if not parquet_path.exists():
                counts["skipped"] += 1
                continue
            features = sequence_features(frames_from_records(read_parquet_records(parquet_path)))
            if features.size == 0:
                counts["skipped"] += 1
                continue
            name = f"asl_{number:06d}.npy"
            np.save(seq_dir / name, features.astype(np.float32))
            record = {"split": split, "label": row["sign"], "signer": participant, "npy": name}
            handle.write(json.dumps(record) + "\n")
            written.append(record)
            counts[split] += 1
    assert_no_signer_leakage(written)
    return counts
