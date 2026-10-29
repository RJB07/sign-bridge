#!/usr/bin/env python3
"""Download Google ASL Signs with a Kaggle token and retrain the gloss model.

Usage:
  export KAGGLE_API_TOKEN=...
  PYTHONPATH=training python training/train_asl_signs.py

`KAGGLE_USERNAME` and `KAGGLE_KEY` work in place of the single token.
The cache is separate from the PopSign image cache. The trainer replaces
public/models/asl-en-sequence.json only when signer-disjoint test top-1 rises.
"""

from __future__ import annotations

import sys
from pathlib import Path

from asl_signs import build_cache, credential_env, download_competition


def main() -> None:
    if credential_env() is None:
        raise SystemExit(
            "No Kaggle credentials in the environment. Set KAGGLE_API_TOKEN, "
            "or both KAGGLE_USERNAME and KAGGLE_KEY. The competition is asl-signs "
            "(CC BY 4.0). This script does not download anything without a token."
        )
    root = Path("training/data/asl-signs")
    cache = Path("training/cache-asl-signs")
    download_competition(root)
    counts = build_cache(root, cache)
    print({"cache": str(cache), **counts}, flush=True)
    sys.argv = [sys.argv[0], "--cache", str(cache)]
    from train_sequence_model import main as train_main

    train_main()


if __name__ == "__main__":
    main()
