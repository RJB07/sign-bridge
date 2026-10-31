#!/usr/bin/env python3
"""Score the 4-member blend on the test split once, after a latency check.

Validation for this weight set is already stored. This file refuses a second
test look once training/metrics/blend4-test.json exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blend_members import load_member, val_rows
from score_blend import logits_for
from train_sequence_model import browser_latency, calibrate_gate, load_index, per_class, ranked_accuracy, write_browser_file

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "training/metrics/blend4-test.json"
SPEC = [
    ("fine", 0, 1.0),
    ("fine", 2, 1.0),
    ("signer", 0, 1.5),
    ("signer", 2, 1.0),
]
VAL_FLOOR = 0.5678
TEST_FLOOR = 0.5676
COMMIT_MS = 130.0


def main() -> None:
    if REPORT.exists():
        print(json.dumps({"refused": "test already scored", "path": str(REPORT)}))
        return
    cache = ROOT / "training/cache"
    rows, labels = val_rows(cache, "index.jsonl")
    torch.set_num_threads(4)
    models = []
    for family, seed, _weight in SPEC:
        directory = ROOT / "training/artifacts" / ("fine192" if family == "fine" else "signer192")
        models.append(load_member(directory / f"member-{seed}.pt", len(labels), 480))
    probe = Path("/tmp/signbridge-blend4-kernel.json")
    write_browser_file(probe, models, labels, "f16", False, 389, "fine", 0.5, 0.1, [0.5] * len(labels), 1.0)
    del models
    latency = browser_latency(probe, runs=8)
    print({"latency": latency}, flush=True)
    if float(latency["commitMs"]) > COMMIT_MS:
        REPORT.write_text(json.dumps({"testReported": False, "reason": "commit slower than 130 ms", "latency": latency}, indent=2), encoding="utf-8")
        return
    val_logits, val_y = logits_for(SPEC, rows, labels, cache)
    val_top1 = ranked_accuracy(val_logits, val_y, 1)
    val_top5 = ranked_accuracy(val_logits, val_y, 5)
    print({"valTop1": round(val_top1, 4), "valTop5": round(val_top5, 4)}, flush=True)
    if val_top1 <= VAL_FLOOR:
        REPORT.write_text(
            json.dumps({"testReported": False, "reason": "validation did not beat 56.8%", "valTop1": round(val_top1, 4), "latency": latency}, indent=2),
            encoding="utf-8",
        )
        return
    gate = calibrate_gate(val_logits, val_y, 0.72)
    index_rows = load_index(cache / "index.jsonl")
    label_to_index = {label: i for i, label in enumerate(labels)}
    test_rows = [row for row in index_rows if row["split"] == "test" and row["label"] in label_to_index]
    REPORT.write_text(json.dumps({"testReported": True, "status": "scoring", "valTop1": round(val_top1, 4)}, indent=2), encoding="utf-8")
    test_logits, test_y = logits_for(SPEC, test_rows, labels, cache)
    temperature = gate["temperature"]
    tempered = test_logits / temperature
    probs = torch.softmax(torch.from_numpy(tempered), dim=1).numpy()
    order = np.argsort(-probs, axis=1)
    pred = order[:, 0]
    conf = probs[np.arange(len(probs)), pred]
    margin = conf - probs[np.arange(len(probs)), order[:, 1]]
    floors = np.array(gate["floors"])
    take = (conf >= floors[pred]) & (margin >= gate["threshold"]["margin"])
    test_top1 = ranked_accuracy(test_logits, test_y, 1)
    test_top5 = ranked_accuracy(test_logits, test_y, 5)
    per = per_class(test_logits, test_y, labels)
    focus = ["hello", "thankyou", "please", "yes", "no", "water", "mom", "dad"]
    result = {
        "testReported": True,
        "members": [f"{family}-{seed}" for family, seed, _weight in SPEC],
        "weights": [weight for _family, _seed, weight in SPEC],
        "latency": latency,
        "valTop1": round(val_top1, 4),
        "valTop5": round(val_top5, 4),
        "temperature": temperature,
        "threshold": gate["threshold"]["threshold"],
        "margin": gate["threshold"]["margin"],
        "classThresholds": [round(float(value), 3) for value in gate["floors"]],
        "testTop1": round(test_top1, 4),
        "testTop5": round(test_top5, 4),
        "testMacroRecall": round(float(np.mean([row["recall"] for row in per if row["support"] > 0])), 4),
        "testAtEmit": {
            "precision": round(float((pred[take] == test_y[take]).mean()), 4) if take.any() else 0.0,
            "coverage": round(float(take.mean()), 4),
        },
        "focus": [row for row in per if row["label"] in focus],
        "previousTestTop1": TEST_FLOOR,
        "beatsShipped": bool(test_top1 > TEST_FLOOR + 1e-4),
        "counts": {"validation": len(rows), "test": len(test_rows)},
    }
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("valTop1", "testTop1", "testTop5", "beatsShipped", "testAtEmit", "latency")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
