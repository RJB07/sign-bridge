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

from features import FEATURE_DIM, SEQ_LEN, augment_sequence, resample_sequence
from sign_transformer import SignTransformer, export_state, forward_logits
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


def train_member(seed: int, n_classes: int, class_weights: np.ndarray, train_rows, label_to_index, cache: Path, val_loader, device, epochs: int, batch_size: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SignTransformer(FEATURE_DIM, n_classes, dropout=0.1).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(
        SequenceDataset(train_rows, label_to_index, cache / "seq", True, seed),
        batch_size=batch_size,
        shuffle=True,
    )
    best_state = None
    best_top1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
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
        row = {
            "seed": seed,
            "epoch": epoch,
            "trainLoss": round(running / max(1, total), 4),
            "trainTop1": round(correct / max(1, total), 4),
            "valTop1": round(val_top1, 4),
            "valTop5": round(val_top5, 4),
        }
        history.append(row)
        print(row, flush=True)
        if val_top1 > best_top1 + 1e-4:
            best_top1 = val_top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    target_all = []
    with torch.no_grad():
        for x, y in loader:
            logits_all.append(model(x.to(device)).cpu().numpy())
            target_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(target_all)


def ensemble_logits(models, loader, device):
    packs = [evaluate(model, loader, device) for model in models]
    target = packs[0][1]
    logits = np.mean([pack[0] for pack in packs], axis=0)
    return logits, target


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
    torch.set_num_threads(4)
    counts = np.bincount([label_to_index[row["label"]] for row in splits["train"]], minlength=len(labels))
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1) / len(labels)).astype(np.float32)
    val_loader = DataLoader(
        SequenceDataset(splits["validation"], label_to_index, cache / "seq", False, 2),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        SequenceDataset(splits["test"], label_to_index, cache / "seq", False, 3),
        batch_size=args.batch_size,
    )
    members = []
    history = []
    for seed in (0, 1):
        model, member_history = train_member(
            seed, len(labels), class_weights, splits["train"], label_to_index, cache, val_loader, device, args.epochs, args.batch_size
        )
        members.append(model)
        history.extend(member_history)

    val_logits, val_y = ensemble_logits(members, val_loader, device)
    test_logits, test_y = ensemble_logits(members, test_loader, device)
    threshold = choose_threshold(val_logits, val_y)
    # Score the chosen gate on the untouched test signers as well.
    probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    order = np.argsort(-probs, axis=1)
    conf = probs[np.arange(len(probs)), order[:, 0]]
    margin = conf - probs[np.arange(len(probs)), order[:, 1]]
    take = (conf >= threshold["threshold"]) & (margin >= threshold["margin"])
    test_at_emit = {
        "threshold": threshold["threshold"],
        "margin": threshold["margin"],
        "precision": round(float((order[:, 0][take] == test_y[take]).mean()), 4) if take.any() else 0.0,
        "coverage": round(float(take.mean()), 4),
    }
    test_per_class = per_class(test_logits, test_y, labels)
    test_top1 = ranked_accuracy(test_logits, test_y, 1)
    test_top5 = ranked_accuracy(test_logits, test_y, min(5, len(labels)))
    macro_recall = float(np.mean([row["recall"] for row in test_per_class if row["support"] > 0])) if test_per_class else 0.0

    exported = [export_state(model) for model in members]
    sample = next(iter(test_loader))[0][:1]
    with torch.no_grad():
        torch_logits = np.mean([model(sample.to(device)).cpu().numpy()[0] for model in members], axis=0)
    ref_parts = [forward_logits(sample[0].numpy(), weights, members[0].heads, len(members[0].blocks)) for weights in exported]
    ref_logits = np.mean(ref_parts, axis=0)
    max_diff = float(np.max(np.abs(torch_logits - ref_logits)))
    if max_diff > 1e-3:
        raise SystemExit(f"NumPy reference drifted from PyTorch by {max_diff}")

    focus = ["hello", "thankyou", "please", "yes", "no", "water", "mom", "dad"]
    focus_rows = [row for row in test_per_class if row["label"] in focus]

    metrics = {
        "task": "isolated ASL gloss to English",
        "language": "ase",
        "outputLanguage": "en",
        "architecture": "ensemble of 2 pre-norm landmark transformers, both hands + pose",
        "dataset": "PopSign ASL v1.0 game images (CC BY 4.0), MediaPipe Holistic landmarks. Same 250 glosses as Google ASL Signs. ASL Citizen and WLASL were not used because their licenses do not allow shipping derived weights.",
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
        "emit": {key: value for key, value in threshold.items() if key != "candidates"},
        "emitCandidates": threshold.get("candidates", []),
        "testAtEmit": test_at_emit,
        "focusGlosses": focus_rows,
        "numpyMaxAbsDiff": max_diff,
        "history": history,
        "perClassTest": test_per_class,
        "notIncluded": {
            "help": "Not one of the 250 PopSign / Google ASL Signs glosses.",
            "aslCitizen": "Microsoft research license forbids distributing the data or modifications.",
            "wlasl": "Video rights are research-only and not redistributable.",
        },
    }
    metrics_dir = Path("training/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "asl_en_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def blob(weights: dict[str, np.ndarray]) -> dict:
        return {
            "weights": {key: encode_f32(value) for key, value in weights.items()},
            "weightShapes": {key: list(value.shape) for key, value in weights.items()},
        }

    model_json = {
        "format": "signbridge-tx-v1",
        "language": "ase",
        "outputLanguage": "en",
        "task": "isolated-asl-gloss-to-english",
        "featureDim": FEATURE_DIM,
        "seqLen": SEQ_LEN,
        "dModel": members[0].d_model,
        "heads": members[0].heads,
        "layers": len(members[0].blocks),
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
            "note": "Same gloss list as the Google Isolated Sign Language Recognition / ASL Signs release.",
        },
        "metrics": {
            "testTop1": metrics["testTop1"],
            "testTop5": metrics["testTop5"],
            "testMacroRecall": metrics["testMacroRecall"],
            "validationTop1": metrics["validationTop1"],
            "counts": metrics["counts"],
            "classes": len(labels),
        },
        "members": [blob(weights) for weights in exported],
    }
    public = Path("public/models")
    public.mkdir(parents=True, exist_ok=True)
    (public / "asl-en-sequence.json").write_text(json.dumps(model_json), encoding="utf-8")
    parity = {
        "input": encode_f32(sample[0].numpy()),
        "logits": encode_f32(ref_logits),
        "inputShape": [SEQ_LEN, FEATURE_DIM],
    }
    (public / "parity.json").write_text(json.dumps(parity), encoding="utf-8")

    correct_by_label: dict[str, dict] = {}
    wanted = ["hello", "thankyou", "please", "yes", "no", "water", "mom", "dad", "book", "bye"]
    with torch.no_grad():
        for row in splits["test"]:
            if len(correct_by_label) >= 8 and row["label"] not in wanted:
                continue
            seq = resample_sequence(np.load(cache / "seq" / row["npy"]), SEQ_LEN)
            parts = [member(torch.from_numpy(seq).unsqueeze(0).to(device)).cpu().numpy()[0] for member in members]
            guess = labels[int(np.mean(parts, axis=0).argmax())]
            if guess == row["label"] and row["label"] not in correct_by_label:
                if row["label"] in wanted or len(correct_by_label) < 8:
                    correct_by_label[row["label"]] = {
                        "label": row["label"],
                        "signer": row["signer"],
                        "features": encode_f32(seq),
                    }
    order = ["hello", "thankyou", "please", "yes", "no", "water", "mom", "dad"]
    clips = [correct_by_label[label] for label in order if label in correct_by_label]
    clips += [correct_by_label[label] for label in labels if label in correct_by_label and label not in order]
    clips = clips[:8]
    (public / "demo-clips.json").write_text(
        json.dumps(
            {
                "note": "Normalized landmark features from held-out PopSign test signers the ensemble classifies correctly. CC BY 4.0. Not for identifying people.",
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )
    summary = {key: metrics[key] for key in ("testTop1", "testTop5", "testMacroRecall", "counts", "classes", "testAtEmit", "focusGlosses")}
    print(json.dumps(summary, indent=2))


MODEL_NOTE = (
    "MediaPipe Holistic landmarker float16, "
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)


if __name__ == "__main__":
    main()
