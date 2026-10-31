#!/usr/bin/env python3
"""Validation-only error analysis for a saved ensemble. Does not read the test split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from features import RICH_FEATURE_DIM
from sign_transformer import SignTransformer
from train_sequence_model import (
    SequenceDataset,
    calibrate_gate,
    choose_threshold,
    ensemble_logits,
    load_index,
    per_class,
    ranked_accuracy,
)


def load_members(directory: Path, n_classes: int, feature_dim: int, d_model: int, layers: int, ff: int, seeds: list[int]):
    models = []
    for seed in seeds:
        path = directory / f"member-{seed}-best.pt"
        if not path.exists():
            path = directory / f"member-{seed}.pt"
        model = SignTransformer(feature_dim, n_classes, d_model=d_model, heads=4, layers=layers, ff=ff, dropout=0.0)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="training/cache")
    parser.add_argument("--index", default="index.jsonl")
    parser.add_argument("--checkpoints", default="training/checkpoints/core64-rich192")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff", type=int, default=480)
    parser.add_argument("--out", default="training/metrics/val-errors.json")
    args = parser.parse_args()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    cache = Path(args.cache)
    rows = load_index(cache / args.index)
    labels = json.loads(Path("public/models/asl-en-sequence.json").read_text(encoding="utf-8"))["labels"]
    label_to_index = {label: index for index, label in enumerate(labels)}
    validation = [row for row in rows if row["split"] == "validation" and row["label"] in label_to_index]
    loader = DataLoader(
        SequenceDataset(validation, label_to_index, cache / "seq-rich", False, 2, layout="rich"),
        batch_size=256,
    )
    device = torch.device("cpu")
    torch.set_num_threads(4)
    models = load_members(Path(args.checkpoints), len(labels), RICH_FEATURE_DIM, args.hidden, args.layers, args.ff, seeds)
    logits, target = ensemble_logits(models, loader, device, tta=False)
    classes = per_class(logits, target, labels)
    worst = sorted(classes, key=lambda row: (row["recall"], -row["support"]))[:25]
    pred = logits.argmax(axis=1)
    pairs: Counter[tuple[str, str]] = Counter()
    for truth, guess in zip(target.tolist(), pred.tolist()):
        if truth != guess:
            pairs[(labels[int(truth)], labels[int(guess)])] += 1
    confusions = [
        {"truth": truth, "guess": guess, "count": count}
        for (truth, guess), count in pairs.most_common(20)
    ]
    gate = calibrate_gate(logits, target, min_precision=0.8)
    current = choose_threshold(logits, target, min_precision=0.72)
    report = {
        "split": "validation",
        "clips": int(len(target)),
        "top1": round(ranked_accuracy(logits, target, 1), 4),
        "top5": round(ranked_accuracy(logits, target, 5), 4),
        "temperature": gate["temperature"],
        "nll": gate["nll"],
        "nllCalibrated": gate["nllCalibrated"],
        "gateUncalibrated": {key: current[key] for key in ("threshold", "margin", "precision", "coverage", "recall", "f1")},
        "gateCalibratedPrecision80": {key: gate["threshold"][key] for key in ("threshold", "margin", "precision", "coverage", "recall", "f1")},
        "worstRecall": worst,
        "topConfusions": confusions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("clips", "top1", "top5", "temperature", "nll", "nllCalibrated", "gateUncalibrated", "gateCalibratedPrecision80")}, indent=2))
    print("worst", ", ".join(f"{row['label']} {row['recall']:.0%} (n={row['support']})" for row in worst[:12]))


if __name__ == "__main__":
    main()
