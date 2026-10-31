#!/usr/bin/env python3
"""Bench blend latency, then score the chosen blend on the test split once.

The candidate is the validation winner that still fits the commit budget.
Re-running this file after blend-test.json exists does not score the test again.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blend_members import load_member, member_logits, val_rows
from train_sequence_model import (
    browser_latency,
    calibrate_gate,
    load_index,
    per_class,
    ranked_accuracy,
    write_browser_file,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "training/metrics/blend-test.json"
VAL_FLOOR = 0.5553
TEST_FLOOR = 0.5667
COMMIT_MS = 130.0

CANDIDATES = [
    {
        "name": "blend4",
        "members": [
            ("fine", 0, 1.0),
            ("fine", 2, 1.0),
            ("signer", 0, 1.5),
            ("signer", 2, 1.0),
        ],
    },
    {
        "name": "blend3",
        "members": [
            ("fine", 0, 1.0),
            ("fine", 2, 1.5),
            ("signer", 1, 2.0),
        ],
    },
    {
        "name": "blend2",
        "members": [
            ("fine", 2, 1.0),
            ("signer", 1, 1.0),
        ],
    },
]


def family_dir(name: str) -> Path:
    return ROOT / "training/artifacts" / ("fine192" if name == "fine" else "signer192")


def family_seq(cache: Path, name: str) -> Path:
    return cache / ("seq-fine" if name == "fine" else "seq-signer")


def weighted(packs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    total = float(sum(weights))
    acc = np.zeros_like(packs[0], dtype=np.float32)
    for pack, weight in zip(packs, weights):
        acc += np.asarray(pack, dtype=np.float32) * (weight / total)
    return acc


def logits_for(spec: list[tuple[str, int, float]], rows: list[dict], labels: list[str], cache: Path) -> tuple[np.ndarray, np.ndarray]:
    packs = []
    target = None
    grouped: dict[str, list] = {"fine": [], "signer": []}
    for family, seed, weight in spec:
        grouped[family].append((seed, weight))
    order = []
    for family, items in grouped.items():
        if not items:
            continue
        # One pass per member. Targets must agree.
        for seed, weight in items:
            model = load_member(family_dir(family) / f"member-{seed}.pt", len(labels), 480)
            logits, y = member_logits(model, rows, {label: i for i, label in enumerate(labels)}, family_seq(cache, family))
            del model
            packs.append(logits)
            order.append(weight)
            if target is None:
                target = y
            elif not np.array_equal(target, y):
                raise SystemExit(f"target mismatch for {family}-{seed}")
            print({"scored": f"{family}-{seed}", "rows": int(len(y))}, flush=True)
    return weighted(packs, order), target


def main() -> None:
    if REPORT.exists():
        print(json.dumps({"refused": "test already scored", "path": str(REPORT)}))
        return
    cache = ROOT / "training/cache"
    rows, labels = val_rows(cache, "index.jsonl")
    torch.set_num_threads(4)
    measured = []
    for candidate in CANDIDATES:
        spec = candidate["members"]
        logits, target = logits_for(spec, rows, labels, cache)
        top1 = ranked_accuracy(logits, target, 1)
        top5 = ranked_accuracy(logits, target, 5)
        models = [load_member(family_dir(family) / f"member-{seed}.pt", len(labels), 480) for family, seed, _weight in spec]
        probe = Path(f"/tmp/signbridge-blend-{candidate['name']}.json")
        write_browser_file(probe, models, labels, "f16", False, 389, "fine", 0.5, 0.1, [0.5] * len(labels), 1.0)
        latency = browser_latency(probe, runs=8)
        row = {
            "name": candidate["name"],
            "members": [f"{family}-{seed}" for family, seed, _weight in spec],
            "weights": [weight for _family, _seed, weight in spec],
            "valTop1": round(top1, 4),
            "valTop5": round(top5, 4),
            "latency": latency,
        }
        measured.append(row)
        print(json.dumps(row), flush=True)
        del models
    legal = [row for row in measured if row["valTop1"] > VAL_FLOOR and float(row["latency"]["commitMs"]) <= COMMIT_MS]
    if not legal:
        payload = {"testReported": False, "reason": "no blend beat validation and the latency budget", "measured": measured}
        REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    chosen = max(legal, key=lambda row: (row["valTop1"], row["valTop5"]))
    spec = next(candidate["members"] for candidate in CANDIDATES if candidate["name"] == chosen["name"])
    index_rows = load_index(cache / "index.jsonl")
    label_to_index = {label: i for i, label in enumerate(labels)}
    test_rows = [row for row in index_rows if row["split"] == "test" and row["label"] in label_to_index]
    # Mark the file before the test forward so a second process cannot start another look.
    REPORT.write_text(json.dumps({"testReported": True, "status": "scoring", "chosen": chosen}, indent=2), encoding="utf-8")
    test_logits, test_y = logits_for(spec, test_rows, labels, cache)
    val_logits, val_y = logits_for(spec, rows, labels, cache)
    gate = calibrate_gate(val_logits, val_y, 0.72)
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
        "chosen": chosen,
        "temperature": temperature,
        "threshold": gate["threshold"]["threshold"],
        "margin": gate["threshold"]["margin"],
        "testTop1": round(test_top1, 4),
        "testTop5": round(test_top5, 4),
        "testMacroRecall": round(float(np.mean([row["recall"] for row in per if row["support"] > 0])), 4),
        "testAtEmit": {
            "precision": round(float((pred[take] == test_y[take]).mean()), 4) if take.any() else 0.0,
            "coverage": round(float(take.mean()), 4),
        },
        "focus": [row for row in per if row["label"] in focus],
        "shippedTestTop1": TEST_FLOOR,
        "beatsShipped": bool(test_top1 > TEST_FLOOR + 1e-4),
        "counts": {"validation": len(rows), "test": len(test_rows)},
        "measured": measured,
    }
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("chosen", "testTop1", "testTop5", "beatsShipped", "testAtEmit")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
