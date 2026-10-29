#!/usr/bin/env python3
"""Download a reproducible subset of PopSign image shards.

The full game split is tens of gigabytes. This script fetches the shard list
used for the committed baseline: early alphabetical shards, which overlap across
the official signer-independent train, validation, and test partitions.

PopSign ASL v1.0 is CC BY 4.0. These files are images, not committed to git.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/sign/popsign-images/resolve/main/data/game"
# Shards chosen so the same English glosses appear in train, validation, and test.
SHARDS = [
    *[f"train-{i:05d}-of-00112.parquet" for i in list(range(0, 17)) + [20, 40, 60, 80, 100]],
    *[f"validation-{i:05d}-of-00030.parquet" for i in [0, 1, 2, 3, 15]],
    *[f"test-{i:05d}-of-00034.parquet" for i in [0, 1, 2, 3, 17]],
]


def download(dest: Path, names: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = dest / name
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"have {name}")
            continue
        url = f"{BASE}/{name}"
        partial = target.with_suffix(target.suffix + ".partial")
        print(f"get {url}")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
        print(f"saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="training/data/popsign/full")
    args = parser.parse_args()
    download(Path(args.dest), SHARDS)


if __name__ == "__main__":
    main()
