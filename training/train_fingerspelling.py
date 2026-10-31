#!/usr/bin/env python3
"""Train the static letter model on a near-duplicate holdout.

sid220/asl-now-fingerspelling (MIT) has no participant id. The same pose often
appears more than once, and a filename hash puts those copies on both sides.
This script keeps a near-duplicate group in one split, trains with palm-scaled
coordinates, fingertip curl, and pose jitter, and scores the held-out groups
once.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from letter_split import group_indices, split_name

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CURLS = ((1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
ROOT = Path(__file__).resolve().parents[1]


def palm_normalized(points: np.ndarray) -> np.ndarray:
    centered = points.astype(np.float32) - points[0]
    scale = float(np.hypot(centered[9, 0], centered[9, 1]))
    if scale < 1e-4:
        scale = 1.0
    return centered / scale


def _curl(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    toward = b - a
    away = c - b
    left = float(np.linalg.norm(toward))
    right = float(np.linalg.norm(away))
    if left < 1e-6 or right < 1e-6:
        return 0.0
    return float(np.dot(toward, away) / (left * right))


def letter_features(points: np.ndarray) -> np.ndarray:
    placed = palm_normalized(points)
    curls = np.array([_curl(placed[a], placed[b], placed[c]) for a, b, c in CURLS], dtype=np.float32)
    return np.concatenate([placed.reshape(-1), curls]).astype(np.float32)


def jitter(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = points.astype(np.float32).copy()
    yaw = float(rng.uniform(-0.35, 0.35))
    pitch = float(rng.uniform(-0.12, 0.12))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x = out[:, 0].copy()
    y = out[:, 1].copy()
    out[:, 0] = cosine * x - sine * y
    out[:, 1] = sine * x + cosine * y
    cp, sp = np.cos(pitch), np.sin(pitch)
    y = out[:, 1].copy()
    z = out[:, 2].copy()
    out[:, 1] = cp * y - sp * z
    out[:, 2] = sp * y + cp * z
    out[:, 0] *= float(rng.uniform(0.9, 1.12))
    out[:, 1] *= float(rng.uniform(0.9, 1.12))
    out += rng.normal(0.0, 0.008, size=out.shape).astype(np.float32)
    return out


def load_examples(root: Path) -> list[dict]:
    examples = []
    for letter in LETTERS:
        folder = root / letter
        names = []
        raws = []
        palms = []
        for path in sorted(folder.glob("*.json")):
            points = np.array([[p["x"], p["y"], p["z"]] for p in json.loads(path.read_text())], dtype=np.float32)
            if points.shape != (21, 3):
                continue
            names.append(path.name)
            raws.append(points)
            palms.append(palm_normalized(points).reshape(-1))
        if not names:
            continue
        for group in group_indices(np.stack(palms)):
            key = min(names[index] for index in group)
            part = split_name(f"{letter}/{key}")
            for index in group:
                examples.append(
                    {
                        "split": part,
                        "letter": letter,
                        "points": raws[index],
                        "name": names[index],
                        "group": f"{letter}/{key}",
                    }
                )
    return examples


class LetterNet(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 26),
        )

    def forward(self, x):
        return self.net(x)


def encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, np.float32).tobytes()).decode()


def main() -> None:
    root = ROOT / "training/data/fingerspelling"
    examples = load_examples(root)
    index = {letter: i for i, letter in enumerate(LETTERS)}
    splits = {name: [row for row in examples if row["split"] == name] for name in ("train", "validation", "test")}
    if min(len(part) for part in splits.values()) < 20:
        raise SystemExit(f"Split too small: {[len(part) for part in splits.values()]}")

    def matrix(part, rng=None):
        rows = []
        for row in part:
            points = jitter(row["points"], rng) if rng is not None else row["points"]
            rows.append(letter_features(points))
        return np.stack(rows).astype(np.float32)

    x_val = torch.tensor(matrix(splits["validation"]))
    y_val = torch.tensor([index[row["letter"]] for row in splits["validation"]], dtype=torch.long)
    x_test = torch.tensor(matrix(splits["test"]))
    y_test = torch.tensor([index[row["letter"]] for row in splits["test"]], dtype=torch.long)
    y_train = torch.tensor([index[row["letter"]] for row in splits["train"]], dtype=torch.long)

    torch.manual_seed(0)
    width = int(x_val.shape[1])
    model = LetterNet(width)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best = None
    best_acc = -1.0
    stale = 0
    rng = np.random.default_rng(0)
    for epoch in range(1, 61):
        model.train()
        x_train = torch.tensor(matrix(splits["train"], rng))
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(x_val).argmax(1) == y_val).float().mean())
        print({"epoch": epoch, "valTop1": round(acc, 4)}, flush=True)
        if acc > best_acc + 1e-4:
            best_acc = acc
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(1).numpy()
        sample_logits = model(x_test[:1]).numpy()[0]
    target = y_test.numpy()
    top1 = float((pred == target).mean())
    per = []
    for i, letter in enumerate(LETTERS):
        mask = target == i
        support = int(mask.sum())
        tp = int(((pred == i) & mask).sum())
        predicted = int((pred == i).sum())
        per.append(
            {
                "label": letter,
                "support": support,
                "precision": round(tp / predicted, 4) if predicted else 0.0,
                "recall": round(tp / support, 4) if support else 0.0,
            }
        )
    linears = [module for module in model.net if isinstance(module, nn.Linear)]
    weights = {}
    for i, layer in enumerate(linears):
        weights[f"fc{i}.weight"] = layer.weight.detach().cpu().numpy()
        weights[f"fc{i}.bias"] = layer.bias.detach().cpu().numpy()
    metrics = {
        "task": "static ASL fingerspelling",
        "license": "MIT sid220/asl-now-fingerspelling",
        "split": (
            "Near-duplicate groups stay in one split (palm-normalized distance 0.35). "
            "Signer ids are not in this dataset, so this is NOT a verified signer holdout. "
            "It does stop the same pose from sitting in train and test. "
            "The older filename-hash number was 92.3% and 47% of that test set had a near-copy in train."
        ),
        "signerIds": False,
        "feature": "wrist-centered palm-scaled hand (63) plus five fingertip curls",
        "counts": {name: len(part) for name, part in splits.items()},
        "groups": {name: len({row["group"] for row in part}) for name, part in splits.items()},
        "validationTop1": round(best_acc, 4),
        "testTop1": round(top1, 4),
        "chanceTop1": round(1 / 26, 4),
        "perClassTest": per,
        "j": next(row for row in per if row["label"] == "J"),
        "z": next(row for row in per if row["label"] == "Z"),
    }
    metrics_path = ROOT / "training/metrics/fingerspelling_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    payload = {
        "format": "signbridge-letters-v1",
        "labels": LETTERS,
        "feature": metrics["feature"],
        "note": metrics["split"],
        "metrics": {
            "testTop1": metrics["testTop1"],
            "validationTop1": metrics["validationTop1"],
            "signerIds": False,
            "j": metrics["j"],
            "z": metrics["z"],
        },
        "weights": {key: encode(value) for key, value in weights.items()},
        "weightShapes": {key: list(value.shape) for key, value in weights.items()},
    }
    out = ROOT / "public/models"
    out.mkdir(parents=True, exist_ok=True)
    (out / "asl-fingerspelling.json").write_text(json.dumps(payload))
    smoke = {
        "input": encode(x_test[0].numpy()),
        "logits": encode(sample_logits.astype(np.float32)),
    }
    (ROOT / "src/recognition/letter-smoke.json").write_text(json.dumps(smoke))
    print(json.dumps({k: metrics[k] for k in ("testTop1", "validationTop1", "counts", "j", "z")}, indent=2))


if __name__ == "__main__":
    main()
