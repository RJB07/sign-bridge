#!/usr/bin/env python3
"""Download every PopSign game shard used for the 250-gloss model.

Existing complete parquet files are skipped. CC BY 4.0 frames from
https://huggingface.co/datasets/sign/popsign-images
"""

from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://huggingface.co/datasets/sign/popsign-images/resolve/main/data/game"
DEST = Path(__file__).resolve().parent / "data" / "popsign" / "full"


def shard_names() -> list[str]:
    names = [f"train-{i:05d}-of-00112.parquet" for i in range(112)]
    names += [f"validation-{i:05d}-of-00030.parquet" for i in range(30)]
    names += [f"test-{i:05d}-of-00034.parquet" for i in range(34)]
    return names


def complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 50_000_000:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == b"PAR1"


def fetch(name: str) -> str:
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / name
    if complete(target):
        return f"skip {name}"
    partial = target.with_suffix(".partial")
    url = f"{BASE}/{name}"
    try:
        urllib.request.urlretrieve(url, partial)
        if not complete(partial) and partial.stat().st_size < 50_000_000:
            return f"small {name} {partial.stat().st_size}"
        # complete() requires PAR1; large files are parquet even before rename
        if partial.stat().st_size < 50_000_000 or partial.read_bytes()[:4] != b"PAR1":
            return f"bad {name}"
        partial.replace(target)
        return f"got {name} {target.stat().st_size / 1e6:.0f}MB"
    except Exception as exc:
        return f"ERR {name} {exc}"


def main() -> None:
    names = shard_names()
    print(f"queued {len(names)}", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, name) for name in names]
        done = 0
        for future in as_completed(futures):
            done += 1
            message = future.result()
            if not message.startswith("skip") or done % 25 == 0:
                print(f"[{done}/{len(names)}] {message}", flush=True)
    print("DOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
