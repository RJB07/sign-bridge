#!/usr/bin/env python3
"""Write the latency-legal blend into the browser model file.

Uses the validation gate and the one test score already stored in
training/metrics/blend-test.json. It does not score the test split again.
Demo clips are held-out examples that this blend already gets right.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blend_members import load_member, val_rows
from features import SEQ_LEN, resample_sequence
from sign_transformer import export_state, forward_logits
from train_sequence_model import encode_f32, load_index, write_browser_file

ROOT = Path(__file__).resolve().parents[1]
SPEC = [
    ("fine", 0, 1.0),
    ("fine", 2, 1.0),
    ("signer", 0, 1.5),
    ("signer", 2, 1.0),
]
ORDER = ["hello", "thankyou", "please", "yes", "no", "water", "mom", "dad"]


def main() -> None:
    report = json.loads((ROOT / "training/metrics/blend4-test.json").read_text(encoding="utf-8"))
    if not report.get("beatsShipped"):
        raise SystemExit("Refusing to export a blend that did not beat the shipped test score")
    cache = ROOT / "training/cache"
    rows, labels = val_rows(cache, "index.jsonl")
    label_to_index = {label: i for i, label in enumerate(labels)}
    torch.set_num_threads(4)
    models = []
    for family, seed, _weight in SPEC:
        directory = ROOT / "training/artifacts" / ("fine192" if family == "fine" else "signer192")
        models.append(load_member(directory / f"member-{seed}.pt", len(labels), 480))
    from score_blend import logits_for

    public = ROOT / "public/models/asl-en-sequence.json"
    floors = report["classThresholds"]
    write_browser_file(
        public,
        models,
        labels,
        "f16",
        False,
        389,
        "fine",
        report["threshold"],
        report["margin"],
        floors,
        report["temperature"],
    )
    payload = json.loads(public.read_text(encoding="utf-8"))
    for member, (family, _seed, weight) in zip(payload["members"], SPEC):
        member["featureLayout"] = family
        member["blendWeight"] = weight
    payload["metrics"] = {
        "testTop1": report["testTop1"],
        "testTop5": report["testTop5"],
        "testMacroRecall": report["testMacroRecall"],
        "validationTop1": report["valTop1"],
        "validationTop5": report["valTop5"],
        "counts": {"train": 40139, "validation": report["counts"]["validation"], "test": report["counts"]["test"]},
        "classes": len(labels),
        "blend": report["members"],
        "commitMs": report["latency"]["commitMs"],
        "previewMs": report["latency"]["previewMs"],
    }
    payload["dataset"] = {
        "name": "PopSign ASL v1.0",
        "license": "CC BY 4.0",
        "huggingface": "https://huggingface.co/datasets/sign/popsign-images",
        "note": "Fine members use shoulder-scaled hand shape. The signer member uses palm-scaled shape. Both are float16.",
    }
    public.write_text(json.dumps(payload), encoding="utf-8")

    rounded = []
    for model in models:
        weights = export_state(model)
        rounded.append({key: np.asarray(value, dtype=np.float16).astype(np.float32) for key, value in weights.items()})
    probe = np.random.default_rng(0).normal(size=(SEQ_LEN, 389)).astype(np.float32)
    total = np.zeros(len(labels), dtype=np.float32)
    weight_sum = 0.0
    for weights, (_family, _seed, weight) in zip(rounded, SPEC):
        total += forward_logits(probe, weights, 4, 3) * weight
        weight_sum += weight
    (ROOT / "public/models/parity.json").write_text(
        json.dumps({"input": encode_f32(probe), "logits": encode_f32(total / weight_sum), "inputShape": [SEQ_LEN, 389]}),
        encoding="utf-8",
    )

    index_rows = load_index(cache / "index.jsonl")
    test_rows = [row for row in index_rows if row["split"] == "test" and row["label"] in label_to_index]
    test_logits, test_y = logits_for(SPEC, test_rows, labels, cache)
    pred = test_logits.argmax(axis=1)
    found: dict[str, dict] = {}
    for index, row in enumerate(test_rows):
        label = row["label"]
        if label not in ORDER or label in found:
            continue
        if int(pred[index]) != int(test_y[index]):
            continue
        fine = resample_sequence(np.load(cache / "seq-fine" / row["npy"]), SEQ_LEN)
        signer = resample_sequence(np.load(cache / "seq-signer" / row["npy"]), SEQ_LEN)
        found[label] = {
            "label": label,
            "signer": row["signer"],
            "features": encode_f32(fine),
            "signerFeatures": encode_f32(signer),
        }
        if len(found) == len(ORDER):
            break
    clips = [found[label] for label in ORDER if label in found]
    if clips[0]["label"] != "hello" or "thankyou" not in found:
        raise SystemExit(f"Replay clips missing hello or thank you: {list(found)}")
    (ROOT / "public/models/demo-clips.json").write_text(
        json.dumps(
            {
                "note": "Held-out PopSign test clips this blend classifies correctly, with shoulder-scaled and palm-scaled features. CC BY 4.0. Not for identifying people.",
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )
    print(json.dumps({"model": str(public), "clips": [clip["label"] for clip in clips], "temperature": report["temperature"], "valTop1": report["valTop1"], "testTop1": report["testTop1"]}, indent=2))


if __name__ == "__main__":
    main()
