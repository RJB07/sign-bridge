#!/usr/bin/env python3
"""Train a static ASL fingerspelling MLP, including J and Z.

Data: sid220/asl-now-fingerspelling (MIT). Each file is one MediaPipe hand.
Participant ids are not in the files, so the holdout is by filename hash.
That can leak a signer across splits. The metrics say so.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize(points: list[dict]) -> np.ndarray:
    arr = np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(arr.shape)
    arr = arr - arr[0]
    scale = float(np.hypot(arr[9, 0], arr[9, 1]))
    if scale < 1e-4:
        scale = 1.0
    return (arr / scale).reshape(-1)


def split_of(name: str) -> str:
    bucket = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def load_rows(root: Path):
    rows = []
    for letter in LETTERS:
        for path in sorted((root / letter).glob("*.json")):
            points = json.loads(path.read_text())
            rows.append((split_of(path.name), letter, normalize(points)))
    return rows


class LetterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 26),
        )

    def forward(self, x):
        return self.net(x)


def encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, np.float32).tobytes()).decode()


def main() -> None:
    root = Path("training/data/fingerspelling")
    rows = load_rows(root)
    index = {letter: i for i, letter in enumerate(LETTERS)}
    splits = {name: [row for row in rows if row[0] == name] for name in ("train", "validation", "test")}

    def tensors(part):
        x = torch.tensor(np.stack([row[2] for row in part]))
        y = torch.tensor([index[row[1]] for row in part], dtype=torch.long)
        return x, y

    x_train, y_train = tensors(splits["train"])
    x_val, y_val = tensors(splits["validation"])
    x_test, y_test = tensors(splits["test"])
    torch.manual_seed(0)
    model = LetterNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    best = None
    best_acc = -1.0
    stale = 0
    for epoch in range(1, 41):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_val).argmax(1)
            acc = float((pred == y_val).float().mean())
        print({"epoch": epoch, "valTop1": round(acc, 4)}, flush=True)
        if acc > best_acc:
            best_acc = acc
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        pred = logits.argmax(1).numpy()
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
    state = model.state_dict()
    # Dropout is absent at eval; export the three linears in order.
    linears = [module for module in model.net if isinstance(module, nn.Linear)]
    weights = {}
    for i, layer in enumerate(linears):
        weights[f"fc{i}.weight"] = layer.weight.detach().cpu().numpy()
        weights[f"fc{i}.bias"] = layer.bias.detach().cpu().numpy()
    metrics = {
        "task": "static ASL fingerspelling",
        "license": "MIT sid220/asl-now-fingerspelling",
        "split": "filename-hash 80/10/10. Signer ids are not in this dataset, so this is NOT signer-independent.",
        "counts": {name: len(part) for name, part in splits.items()},
        "testTop1": round(top1, 4),
        "chanceTop1": round(1 / 26, 4),
        "perClassTest": per,
        "j": next(row for row in per if row["label"] == "J"),
        "z": next(row for row in per if row["label"] == "Z"),
    }
    Path("training/metrics").mkdir(exist_ok=True)
    Path("training/metrics/fingerspelling_metrics.json").write_text(json.dumps(metrics, indent=2))
    payload = {
        "format": "signbridge-letters-v1",
        "labels": LETTERS,
        "feature": "wrist-centered hand / palm scale, 63 floats, static frame",
        "note": metrics["split"],
        "metrics": {"testTop1": metrics["testTop1"], "j": metrics["j"], "z": metrics["z"]},
        "weights": {key: encode(value) for key, value in weights.items()},
        "weightShapes": {key: list(value.shape) for key, value in weights.items()},
    }
    out = Path("public/models")
    out.mkdir(parents=True, exist_ok=True)
    (out / "asl-fingerspelling.json").write_text(json.dumps(payload))
    print(json.dumps({k: metrics[k] for k in ("testTop1", "counts", "j", "z")}, indent=2))


if __name__ == "__main__":
    main()
