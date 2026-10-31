#!/usr/bin/env python3
"""Build the 294-d face-relative sequences from cached raw landmarks.

The extractor stores MediaPipe points in ``raw/`` and the legacy 273-d vector
in ``seq/``. This pass does not download or run MediaPipe again. Validation and
test clips stay the files already in the cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import RICH_FEATURE_DIM, frames_from_raw, sequence_features_rich


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="training/cache")
    parser.add_argument("--index", default="index.jsonl")
    args = parser.parse_args()
    cache = Path(args.cache)
    dest = cache / "seq-rich"
    dest.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (cache / args.index).read_text(encoding="utf-8").splitlines() if line.strip()]
    written = 0
    skipped = 0
    missing = 0
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
        features = sequence_features_rich(frames_from_raw(np.load(raw_path)))
        if features.shape[0] < 2 or features.shape[1] != RICH_FEATURE_DIM:
            continue
        np.save(target, features.astype(np.float32))
        written += 1
        if written % 2000 == 0:
            print({"written": written, "skipped": skipped, "missing": missing}, flush=True)
    print({"written": written, "skipped": skipped, "missing": missing, "dest": str(dest)}, flush=True)


if __name__ == "__main__":
    main()
