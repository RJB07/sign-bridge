#!/usr/bin/env python3
"""Build 389-d signer sequences: fine features with palm-sized hand shape."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from features import FINE_FEATURE_DIM, frames_from_raw, sequence_features_signer


def _one(job: tuple[str, str]) -> str:
    raw_path, target = job
    features = sequence_features_signer(frames_from_raw(np.load(raw_path)))
    if features.shape[0] < 2 or features.shape[1] != FINE_FEATURE_DIM:
        return "short"
    np.save(target, features.astype(np.float32))
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="training/cache")
    parser.add_argument("--index", default="index.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cache = Path(args.cache)
    dest = cache / "seq-signer"
    dest.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (cache / args.index).read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = []
    skipped = missing = 0
    for row in rows:
        name = row["npy"]
        target = dest / name
        if target.exists():
            skipped += 1
            continue
        raw_path = cache / "raw" / name
        if not raw_path.exists():
            missing += 1
            continue
        jobs.append((str(raw_path), str(target)))
    written = short = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for index, status in enumerate(pool.map(_one, jobs, chunksize=32), start=1):
                if status == "written":
                    written += 1
                else:
                    short += 1
                if index % 4000 == 0:
                    print({"seen": index, "written": written, "short": short, "skipped": skipped}, flush=True)
    print({"written": written, "short": short, "skipped": skipped, "missing": missing, "dest": str(dest)}, flush=True)


if __name__ == "__main__":
    main()
