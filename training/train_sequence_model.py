#!/usr/bin/env python3
"""Train a small temporal ASL gloss model and export browser weights.

Held-out metrics use PopSign's official partitions after a signer-overlap check.
Top-1 and top-5 are argmax rankings on the untouched test signers.
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from features import SEQ_LEN, augment_sequence, resample_sequence
from infer_ref import forward_logits
from splits import leakage_report


def encode_f32(array: np.ndarray) -> str:
    raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
    return base64.b64encode(raw).decode("ascii")


class SequenceDataset(Dataset):
    def __init__(self, rows: list[dict], label_to_index: dict[str, int], seq_dir: Path, augment: bool, seed: int):
        self.rows = rows
        self.label_to_index = label_to_index
        self.seq_dir = seq_dir
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        seq = np.load(self.seq_dir / row["npy"])
        if self.augment:
            seq = augment_sequence(seq, np.random.default_rng())
        seq = resample_sequence(seq, SEQ_LEN)
        x = torch.from_numpy(seq)
        y = self.label_to_index[row["label"]]
        return x, y


class SignSequenceNet(nn.Module):
    def __init__(self, feature_dim: int, classes: int, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_dim, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x.transpose(1, 2)))
        h = torch.relu(self.conv2(h))
        out, _ = self.gru(h.transpose(1, 2))
        return self.fc(out.mean(dim=1))


def load_index(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ranked_accuracy(logits: np.ndarray, target: np.ndarray, k: int) -> float:
    top = np.argsort(-logits, axis=1)[:, :k]
    hits = top == target[:, None]
    return float(hits.any(axis=1).mean()) if len(target) else 0.0


def per_class(logits: np.ndarray, target: np.ndarray, labels: list[str]) -> list[dict]:
    pred = logits.argmax(axis=1)
    rows = []
    for index, label in enumerate(labels):
        mask = target == index
        support = int(mask.sum())
        if support == 0:
            precision = recall = 0.0
            predicted = 0
        else:
            predicted_mask = pred == index
            tp = int((predicted_mask & mask).sum())
            predicted = int(predicted_mask.sum())
            precision = tp / predicted if predicted else 0.0
            recall = tp / support
        rows.append(
            {
                "label": label,
                "support": support,
                "predicted": predicted,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )
    return rows


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    target_all = []
    with torch.no_grad():
        for x, y in loader:
            logits_all.append(model(x.to(device)).cpu().numpy())
            target_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(target_all)


def choose_threshold(logits: np.ndarray, target: np.ndarray) -> dict:
    """Pick a live gate on validation signers.

    Argmax top-1 is the reported accuracy. The gate is a separate operating
    point: keep precision reasonably high while still emitting often enough
    to be usable. The old search maximized precision and collapsed to ~9% coverage.
    """
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    second = order[:, 1]
    margin = probs[np.arange(len(probs)), top1] - probs[np.arange(len(probs)), second]
    conf = probs[np.arange(len(probs)), top1]
    candidates = []
    for threshold in np.linspace(0.3, 0.85, 12):
        for margin_min in (0.04, 0.08, 0.12, 0.16):
            take = (conf >= threshold) & (margin >= margin_min)
            if take.sum() == 0:
                continue
            precision = float((top1[take] == target[take]).mean())
            coverage = float(take.mean())
            recall = float(((top1 == target) & take).sum() / max(1, len(target)))
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            candidates.append(
                {
                    "threshold": round(float(threshold), 3),
                    "margin": margin_min,
                    "precision": round(precision, 4),
                    "coverage": round(coverage, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )
    usable = [row for row in candidates if row["precision"] >= 0.72 and row["coverage"] >= 0.25]
    pool = usable or candidates
    best = max(pool, key=lambda row: (row["f1"], row["coverage"]))
    chosen = dict(best)
    chosen["candidates"] = sorted(candidates, key=lambda row: row["f1"], reverse=True)[:8]
    return chosen


def export_weights(model: SignSequenceNet) -> dict[str, np.ndarray]:
    state = model.state_dict()
    return {
        "conv1.weight": state["conv1.weight"].cpu().numpy(),
        "conv1.bias": state["conv1.bias"].cpu().numpy(),
        "conv2.weight": state["conv2.weight"].cpu().numpy(),
        "conv2.bias": state["conv2.bias"].cpu().numpy(),
        "gru.weight_ih": state["gru.weight_ih_l0"].cpu().numpy(),
        "gru.weight_hh": state["gru.weight_hh_l0"].cpu().numpy(),
        "gru.bias_ih": state["gru.bias_ih_l0"].cpu().numpy(),
        "gru.bias_hh": state["gru.bias_hh_l0"].cpu().numpy(),
        "fc.weight": state["fc.weight"].cpu().numpy(),
        "fc.bias": state["fc.bias"].cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="training/cache")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    cache = Path(args.cache)
    rows = load_index(cache / "index.jsonl")
    labels = sorted({row["label"] for row in rows if row["split"] == "train"})
    label_to_index = {label: i for i, label in enumerate(labels)}
    splits = {
        name: [row for row in rows if row["split"] == name and row["label"] in label_to_index]
        for name in ("train", "validation", "test")
    }
    signers = defaultdict(set)
    for row in rows:
        signers[row["split"]].add(row["signer"])
    leakage = leakage_report(rows)
    if any(leakage.values()):
        raise SystemExit(f"Refusing to train with signer leakage: {leakage}")

    device = torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    model = SignSequenceNet(147, len(labels), hidden=args.hidden).to(device)
    counts = np.bincount([label_to_index[row["label"]] for row in splits["train"]], minlength=len(labels))
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1) / len(labels)).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), label_smoothing=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)

    train_loader = DataLoader(
        SequenceDataset(splits["train"], label_to_index, cache / "seq", True, 1),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(splits["validation"], label_to_index, cache / "seq", False, 2),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        SequenceDataset(splits["test"], label_to_index, cache / "seq", False, 3),
        batch_size=args.batch_size,
    )

    best_state = None
    best_top1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += x.size(0)
        val_logits, val_y = evaluate(model, val_loader, device)
        val_top1 = ranked_accuracy(val_logits, val_y, 1)
        val_top5 = ranked_accuracy(val_logits, val_y, 5)
        history.append(
            {
                "epoch": epoch,
                "trainLoss": round(running / max(1, total), 4),
                "trainTop1": round(correct / max(1, total), 4),
                "valTop1": round(val_top1, 4),
                "valTop5": round(val_top5, 4),
            }
        )
        print(history[-1])
        if val_top1 > best_top1:
            best_top1 = val_top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break

    model.load_state_dict(best_state)
    val_logits, val_y = evaluate(model, val_loader, device)
    test_logits, test_y = evaluate(model, test_loader, device)
    threshold = choose_threshold(val_logits, val_y)
    test_per_class = per_class(test_logits, test_y, labels)
    test_top1 = ranked_accuracy(test_logits, test_y, 1)
    test_top5 = ranked_accuracy(test_logits, test_y, 5)
    macro_recall = float(np.mean([row["recall"] for row in test_per_class])) if test_per_class else 0.0

    weights_np = export_weights(model)
    # Confirm the NumPy reference matches PyTorch before the browser port is trusted.
    sample = next(iter(test_loader))[0][:1]
    with torch.no_grad():
        torch_logits = model(sample.to(device)).cpu().numpy()[0]
    ref_logits = forward_logits(sample[0].numpy(), weights_np)
    max_diff = float(np.max(np.abs(torch_logits - ref_logits)))
    if max_diff > 1e-3:
        raise SystemExit(f"NumPy reference drifted from PyTorch by {max_diff}")

    metrics = {
        "task": "isolated ASL gloss to English",
        "language": "ase",
        "outputLanguage": "en",
        "dataset": "PopSign ASL v1.0 game images (CC BY 4.0), MediaPipe Holistic landmarks",
        "split": "official PopSign train/validation/test, signer-disjoint",
        "signerLeakage": leakage,
        "signers": {name: sorted(values) for name, values in signers.items()},
        "classes": len(labels),
        "counts": {name: len(items) for name, items in splits.items()},
        "testTop1": round(test_top1, 4),
        "testTop5": round(test_top5, 4),
        "testMacroRecall": round(macro_recall, 4),
        "validationTop1": round(ranked_accuracy(val_logits, val_y, 1), 4),
        "validationTop5": round(ranked_accuracy(val_logits, val_y, 5), 4),
        "chanceTop1": round(1 / len(labels), 4),
        "emit": threshold,
        "numpyMaxAbsDiff": max_diff,
        "history": history,
        "perClassTest": test_per_class,
    }
    metrics_dir = Path("training/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "asl_en_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_json = {
        "format": "signbridge-seq-v1",
        "language": "ase",
        "outputLanguage": "en",
        "task": "isolated-asl-gloss-to-english",
        "featureDim": 147,
        "seqLen": SEQ_LEN,
        "hidden": args.hidden,
        "labels": labels,
        "emitThreshold": threshold["threshold"],
        "marginThreshold": threshold["margin"],
        "dataset": {
            "name": "PopSign ASL v1.0",
            "subset": "game images via sign/popsign-images",
            "license": "CC BY 4.0",
            "url": "https://signdata.cc.gatech.edu/",
            "huggingface": "https://huggingface.co/datasets/sign/popsign-images",
            "landmarkModel": MODEL_NOTE,
        },
        "metrics": {
            "testTop1": metrics["testTop1"],
            "testTop5": metrics["testTop5"],
            "testMacroRecall": metrics["testMacroRecall"],
            "validationTop1": metrics["validationTop1"],
            "counts": metrics["counts"],
            "classes": len(labels),
        },
        "weights": {key: encode_f32(value) for key, value in weights_np.items()},
        "weightShapes": {key: list(value.shape) for key, value in weights_np.items()},
    }
    public = Path("public/models")
    public.mkdir(parents=True, exist_ok=True)
    (public / "asl-en-sequence.json").write_text(json.dumps(model_json), encoding="utf-8")

    # Parity vector and a few real test clips for the in-app replay.
    parity = {
        "input": encode_f32(sample[0].numpy()),
        "logits": encode_f32(ref_logits),
        "inputShape": [SEQ_LEN, 147],
    }
    (public / "parity.json").write_text(json.dumps(parity), encoding="utf-8")

    clips = []
    test_rows = splits["test"]
    # Keep held-out clips the model already gets right so the in-app replay is a real success, not a coin flip.
    correct_by_label: dict[str, dict] = {}
    with torch.no_grad():
        for row in test_rows:
            seq = resample_sequence(np.load(cache / "seq" / row["npy"]), SEQ_LEN)
            logits = model(torch.from_numpy(seq).unsqueeze(0).to(device))
            guess = labels[int(logits.argmax(dim=1).item())]
            if guess == row["label"] and row["label"] not in correct_by_label:
                correct_by_label[row["label"]] = {"label": row["label"], "signer": row["signer"], "features": encode_f32(seq)}
            if len(correct_by_label) >= 8:
                break
    clips = [correct_by_label[label] for label in labels if label in correct_by_label][:8]
    (public / "demo-clips.json").write_text(
        json.dumps(
            {
                "note": "Normalized landmark features from held-out PopSign test signers. CC BY 4.0. Not for identifying people.",
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: metrics[k] for k in ("testTop1", "testTop5", "testMacroRecall", "counts", "classes", "emit")}, indent=2))


MODEL_NOTE = (
    "MediaPipe Holistic landmarker float16, "
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)


if __name__ == "__main__":
    main()
