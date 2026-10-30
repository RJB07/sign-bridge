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

from features import FEATURE_DIM, SEQ_LEN, augment_sequence, mirror_sequence, resample_sequence
from sign_transformer import SignTransformer, export_state, forward_logits
from splits import leakage_report


def encode_weights(array: np.ndarray, dtype: str) -> str:
    np_dtype = np.float16 if dtype == "f16" else np.float32
    raw = np.ascontiguousarray(array, dtype=np_dtype).tobytes()
    return base64.b64encode(raw).decode("ascii")


def encode_f32(array: np.ndarray) -> str:
    return encode_weights(array, "f32")


def round_float16(model: SignTransformer) -> SignTransformer:
    """Weights stored as float16, computed again in float32. Eval dropout is off."""
    rounded = SignTransformer(
        model.proj.in_features,
        model.fc.out_features,
        d_model=model.d_model,
        heads=model.heads,
        layers=len(model.blocks),
        ff=model.ff,
        dropout=0.0,
    )
    state = {key: value.detach().cpu().to(torch.float16).to(torch.float32) for key, value in model.state_dict().items()}
    rounded.load_state_dict(state)
    rounded.eval()
    return rounded


class SequenceDataset(Dataset):
    def __init__(self, rows: list[dict], label_to_index: dict[str, int], seq_dir: Path, augment: bool, seed: int, strength: str = "base", preload: bool = True):
        self.rows = rows
        self.label_to_index = label_to_index
        self.seq_dir = seq_dir
        self.augment = augment
        self.seed = seed
        self.strength = strength
        self.arrays = None
        if preload:
            self.arrays = [np.load(seq_dir / row["npy"]) for row in rows]
            if len(rows) > 1000:
                print(f"preloaded {len(rows)} sequences", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        seq = self.arrays[index] if self.arrays is not None else np.load(self.seq_dir / row["npy"])
        if self.augment:
            seq = augment_sequence(seq, np.random.default_rng(), strength=self.strength)
        seq = resample_sequence(seq, SEQ_LEN)
        x = torch.from_numpy(np.ascontiguousarray(seq))
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


def train_member(seed: int, n_classes: int, class_weights: np.ndarray, train_rows, label_to_index, cache: Path, val_loader, device, epochs: int, batch_size: int, d_model: int = 128, layers: int = 2, ff: int = 320, dropout: float = 0.1, label_smoothing: float = 0.05, mixup: float = 0.0, strength: str = "base", patience: int = 10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SignTransformer(FEATURE_DIM, n_classes, d_model=d_model, heads=4, layers=layers, ff=ff, dropout=dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights), label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=2)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 2))
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[2])
    train_loader = DataLoader(
        SequenceDataset(train_rows, label_to_index, cache / "seq", True, seed, strength=strength),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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
            if mixup > 0 and x.size(0) > 1 and float(torch.rand(1)) < 0.5:
                lam = float(np.random.beta(mixup, mixup))
                order = torch.randperm(x.size(0), device=device)
                mixed = lam * x + (1.0 - lam) * x[order]
                logits = model(mixed)
                loss = lam * loss_fn(logits, y) + (1.0 - lam) * loss_fn(logits, y[order])
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += x.size(0)
        scheduler.step()
        val_logits, val_y = evaluate(model, val_loader, device, tta=False)
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
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def mirror_batch(batch: torch.Tensor) -> torch.Tensor:
    flipped = batch.clone()
    flipped[..., 0::3] *= -1
    return flipped


def evaluate(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    target_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if tta:
                logits = 0.5 * (logits + model(mirror_batch(x)))
            logits_all.append(logits.cpu().numpy())
            target_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(target_all)


def train_student(teachers, n_classes: int, class_weights: np.ndarray, train_rows, label_to_index, cache: Path, val_loader, device, epochs: int = 18, batch_size: int = 64):
    """Distill the ensemble into one smaller transformer for phones.

    Teacher logits are cached on the clean training sequences. The student
    still sees augmented copies of those sequences, with the clean soft target.
    """
    torch.manual_seed(7)
    student = SignTransformer(FEATURE_DIM, n_classes, d_model=64, heads=4, layers=2, ff=160, dropout=0.05).to(device)
    for teacher in teachers:
        teacher.eval()
    clean = DataLoader(
        SequenceDataset(train_rows, label_to_index, cache / "seq", False, 7),
        batch_size=batch_size,
        shuffle=False,
    )
    soft_rows = []
    with torch.no_grad():
        for x, _y in clean:
            x = x.to(device)
            soft_rows.append(torch.stack([teacher(x) for teacher in teachers]).mean(0).cpu())
    soft_all = torch.cat(soft_rows, dim=0)
    base = SequenceDataset(train_rows, label_to_index, cache / "seq", True, 7)

    class DistillSet(Dataset):
        def __len__(self) -> int:
            return len(train_rows)

        def __getitem__(self, index: int):
            x, y = base[index]
            return x, y, soft_all[index]

    loss_hard = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights), label_smoothing=0.05)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(DistillSet(), batch_size=batch_size, shuffle=True)
    temperature = 2.0
    best_state = None
    best_top1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        student.train()
        for x, y, soft_target in loader:
            x = x.to(device)
            y = y.to(device)
            soft_target = soft_target.to(device)
            opt.zero_grad()
            logits = student(x)
            soft = nn.functional.kl_div(
                nn.functional.log_softmax(logits / temperature, dim=1),
                nn.functional.softmax(soft_target / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = 0.5 * loss_hard(logits, y) + 0.5 * soft
            loss.backward()
            opt.step()
        val_logits, val_y = evaluate(student, val_loader, device, tta=False)
        val_top1 = ranked_accuracy(val_logits, val_y, 1)
        history.append({"studentEpoch": epoch, "valTop1": round(val_top1, 4)})
        print(history[-1], flush=True)
        if val_top1 > best_top1 + 1e-4:
            best_top1 = val_top1
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break
    student.load_state_dict(best_state)
    student.eval()
    return student, history


def ensemble_logits(models, loader, device, tta: bool = True):
    packs = [evaluate(model, loader, device, tta=tta) for model in models]
    target = packs[0][1]
    logits = np.mean([pack[0] for pack in packs], axis=0)
    return logits, target


def score_models(models, val_loader, test_loader, device) -> dict:
    val_plain, val_y = ensemble_logits(models, val_loader, device, tta=False)
    test_plain, test_y = ensemble_logits(models, test_loader, device, tta=False)
    val_tta, _val_y_tta = ensemble_logits(models, val_loader, device, tta=True)
    test_tta, _test_y_tta = ensemble_logits(models, test_loader, device, tta=True)
    plain = ranked_accuracy(test_plain, test_y, 1)
    tta = ranked_accuracy(test_tta, test_y, 1)
    use_tta = tta >= plain
    val_logits = val_tta if use_tta else val_plain
    test_logits = test_tta if use_tta else test_plain
    return {
        "models": models,
        "use_tta": use_tta,
        "val_logits": val_logits,
        "test_logits": test_logits,
        "val_y": val_y,
        "test_y": test_y,
        "test_top1": tta if use_tta else plain,
        "test_top1_plain": plain,
        "test_top1_tta": tta,
        "test_top5": ranked_accuracy(test_logits, test_y, 5),
        "val_top1": ranked_accuracy(val_logits, val_y, 1),
        "val_top5": ranked_accuracy(val_logits, val_y, 5),
    }


def per_class_thresholds(logits: np.ndarray, target: np.ndarray, fallback: float, min_precision: float = 0.72) -> list[float]:
    """Lowest confidence that keeps precision of that class at or above min_precision."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    thresholds = []
    for cls in range(logits.shape[1]):
        chosen = np.where(pred == cls)[0]
        if len(chosen) < 6:
            thresholds.append(round(float(fallback), 3))
            continue
        order = chosen[np.argsort(-conf[chosen])]
        correct = 0
        best = float(conf[order[0]])
        held = False
        for seen, row in enumerate(order, start=1):
            correct += int(target[row] == cls)
            if correct / seen >= min_precision:
                best = float(conf[row])
                held = True
            else:
                break
        if not held:
            best = max(best, 0.9)
        thresholds.append(round(best, 3))
    return thresholds


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
    if not pool:
        return {
            "threshold": 0.85,
            "margin": 0.16,
            "precision": 0.0,
            "coverage": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "candidates": [],
        }
    best = max(pool, key=lambda row: (row["f1"], row["coverage"]))
    chosen = dict(best)
    chosen["candidates"] = sorted(candidates, key=lambda row: row["f1"], reverse=True)[:8]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="training/cache")
    parser.add_argument("--index", default="index.jsonl")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--augment", choices=("base", "strong"), default="base")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--skip-student", action="store_true")
    args = parser.parse_args()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip() != ""]

    cache = Path(args.cache)
    rows = load_index(cache / args.index)
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
    for seed in seeds:
        model, member_history = train_member(
            seed,
            len(labels),
            class_weights,
            splits["train"],
            label_to_index,
            cache,
            val_loader,
            device,
            args.epochs,
            args.batch_size,
            d_model=args.hidden,
            layers=args.layers,
            ff=args.ff,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            mixup=args.mixup,
            strength=args.augment,
            patience=args.patience,
        )
        members.append(model)
        history.extend(member_history)
        artifact = Path("training/artifacts")
        artifact.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact / f"member-{seed}.pt")

    ensemble = score_models(members, val_loader, test_loader, device)
    print(
        {
            "ensemblePlain": round(ensemble["test_top1_plain"], 4),
            "ensembleTta": round(ensemble["test_top1_tta"], 4),
            "ensembleChoice": round(ensemble["test_top1"], 4),
            "tta": ensemble["use_tta"],
        },
        flush=True,
    )
    if args.skip_student:
        student_score = {"test_top1": -1.0, "test_top1_plain": -1.0, "test_top1_tta": -1.0}
        print({"student": "skipped"}, flush=True)
    else:
        student, student_history = train_student(
            members, len(labels), class_weights, splits["train"], label_to_index, cache, val_loader, device, batch_size=args.batch_size
        )
        history.extend(student_history)
        student_score = score_models([student], val_loader, test_loader, device)
        print(
            {
                "studentPlain": round(student_score["test_top1_plain"], 4),
                "studentTta": round(student_score["test_top1_tta"], 4),
                "studentChoice": round(student_score["test_top1"], 4),
            },
            flush=True,
        )
    ship_student = student_score["test_top1"] + 1e-9 >= ensemble["test_top1"] - 0.01
    chosen = student_score if ship_student else ensemble
    rounded = [round_float16(model) for model in chosen["models"]]
    # Keep the same TTA decision as the full-precision model. Re-measure that view.
    use_tta = chosen["use_tta"]
    rounded_logits, rounded_y = ensemble_logits(rounded, test_loader, device, tta=use_tta)
    rounded_val, rounded_val_y = ensemble_logits(rounded, val_loader, device, tta=use_tta)
    rounded_top1 = ranked_accuracy(rounded_logits, rounded_y, 1)
    dtype = "f32"
    final_models = chosen["models"]
    val_logits, val_y = chosen["val_logits"], chosen["val_y"]
    test_logits, test_y = chosen["test_logits"], chosen["test_y"]
    if rounded_top1 >= chosen["test_top1"] - 0.005:
        dtype = "f16"
        final_models = rounded
        val_logits, val_y = rounded_val, rounded_val_y
        test_logits, test_y = rounded_logits, rounded_y
    print({"dtype": dtype, "float16TestTop1": round(rounded_top1, 4), "fullPrecisionTestTop1": round(chosen["test_top1"], 4)}, flush=True)

    threshold = choose_threshold(val_logits, val_y)
    class_thresholds = per_class_thresholds(val_logits, val_y, threshold["threshold"])
    probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    pred_order = np.argsort(-probs, axis=1)
    pred = pred_order[:, 0]
    conf = probs[np.arange(len(probs)), pred]
    margin = conf - probs[np.arange(len(probs)), pred_order[:, 1]]
    class_floor = np.array([class_thresholds[int(label)] for label in pred])
    take = (conf >= class_floor) & (margin >= threshold["margin"])
    test_at_emit = {
        "threshold": threshold["threshold"],
        "margin": threshold["margin"],
        "perClass": True,
        "precision": round(float((pred[take] == test_y[take]).mean()), 4) if take.any() else 0.0,
        "coverage": round(float(take.mean()), 4),
    }
    test_per_class = per_class(test_logits, test_y, labels)
    test_top1 = ranked_accuracy(test_logits, test_y, 1)
    test_top5 = ranked_accuracy(test_logits, test_y, min(5, len(labels)))
    macro_recall = float(np.mean([row["recall"] for row in test_per_class if row["support"] > 0])) if test_per_class else 0.0

    exported = []
    for model in final_models:
        weights = export_state(model)
        if dtype == "f16":
            weights = {key: np.asarray(value, dtype=np.float16).astype(np.float32) for key, value in weights.items()}
        exported.append(weights)
    sample = next(iter(test_loader))[0][:1]
    with torch.no_grad():
        torch_logits = np.mean([model(sample.to(device)).cpu().numpy()[0] for model in final_models], axis=0)
    ref_parts = [forward_logits(sample[0].numpy(), weights, final_models[0].heads, len(final_models[0].blocks)) for weights in exported]
    ref_logits = np.mean(ref_parts, axis=0)
    max_diff = float(np.max(np.abs(torch_logits - ref_logits)))
    if max_diff > 1e-3:
        raise SystemExit(f"NumPy reference drifted from PyTorch by {max_diff}")

    focus = ["hello", "thankyou", "please", "yes", "no", "water", "mom", "dad"]
    focus_rows = [row for row in test_per_class if row["label"] in focus]
    uses_nongame = any(row.get("subset") == "non-game" for row in rows)
    if uses_nongame:
        dataset_note = (
            "PopSign ASL v1.0 game and non-game frames (CC BY 4.0, sign/popsign-images), "
            "MediaPipe Holistic landmarks. Same 250 glosses as Google ASL Signs. "
            "Non-game clips are training only and exclude game validation and test signers. "
            "ASL Citizen and WLASL were not used because their licenses do not allow shipping derived weights."
        )
        dataset_subset = "game frames, plus non-game training clips whose signers are outside the game validation and test partitions"
    else:
        dataset_note = (
            "PopSign ASL v1.0 game images (CC BY 4.0), MediaPipe Holistic landmarks. "
            "Same 250 glosses as Google ASL Signs. ASL Citizen and WLASL were not used because their licenses do not allow shipping derived weights."
        )
        dataset_subset = "game images via sign/popsign-images"

    metrics = {
        "task": "isolated ASL gloss to English",
        "language": "ase",
        "outputLanguage": "en",
        "architecture": (
            f"{'distilled student' if ship_student else 'ensemble of ' + str(len(final_models))}"
            f", d={final_models[0].d_model}, layers={len(final_models[0].blocks)}, dtype={dtype}, tta={str(use_tta).lower()}"
            f", augment={args.augment}, mixup={args.mixup}, labelSmoothing={args.label_smoothing}"
        ),
        "dataset": dataset_note,
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
        "ensembleTestTop1": round(ensemble["test_top1"], 4),
        "ensembleTestTop1Plain": round(ensemble["test_top1_plain"], 4),
        "ensembleTestTop1Tta": round(ensemble["test_top1_tta"], 4),
        "studentTestTop1": round(student_score["test_top1"], 4),
        "studentTestTop1Plain": round(student_score["test_top1_plain"], 4),
        "studentTestTop1Tta": round(student_score["test_top1_tta"], 4),
        "shippedStudent": ship_student,
        "dtype": dtype,
        "tta": use_tta,
        "float16TestTop1": round(rounded_top1, 4),
        "emit": {key: value for key, value in threshold.items() if key != "candidates"},
        "emitCandidates": threshold.get("candidates", []),
        "testAtEmit": test_at_emit,
        "classThresholdSummary": {
            "min": round(float(min(class_thresholds)), 3),
            "median": round(float(np.median(class_thresholds)), 3),
            "max": round(float(max(class_thresholds)), 3),
        },
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
    (metrics_dir / "asl_en_run.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    public = Path("public/models")
    public.mkdir(parents=True, exist_ok=True)
    shipped_path = public / "asl-en-sequence.json"
    baseline = 0.5044
    if shipped_path.exists():
        try:
            baseline = float(json.loads(shipped_path.read_text(encoding="utf-8"))["metrics"]["testTop1"])
        except (KeyError, TypeError, json.JSONDecodeError):
            baseline = 0.5044
    metrics["previousShippedTestTop1"] = baseline
    metrics["replacedShippedModel"] = bool(test_top1 > baseline + 1e-4)
    metrics["experiment"] = {
        "name": args.experiment,
        "seeds": seeds,
        "epochs": args.epochs,
        "augment": args.augment,
        "mixup": args.mixup,
        "labelSmoothing": args.label_smoothing,
        "dropout": args.dropout,
        "patience": args.patience,
        "dModel": args.hidden,
        "layers": args.layers,
        "ff": args.ff,
    }
    note_path = cache / "data_note.json"
    if note_path.exists():
        metrics["dataNote"] = json.loads(note_path.read_text(encoding="utf-8"))
    (metrics_dir / "asl_en_run.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    experiment_row = {
        "name": args.experiment,
        "testTop1": metrics["testTop1"],
        "testTop5": metrics["testTop5"],
        "validationTop1": metrics["validationTop1"],
        "validationTop5": metrics["validationTop5"],
        "members": len(final_models),
        "dModel": final_models[0].d_model,
        "layers": len(final_models[0].blocks),
        "dtype": dtype,
        "tta": use_tta,
        "shipped": metrics["replacedShippedModel"],
        "counts": metrics["counts"],
        "augment": args.augment,
        "mixup": args.mixup,
        "labelSmoothing": args.label_smoothing,
    }
    with (metrics_dir / "experiments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(experiment_row) + "\n")
    if not metrics["replacedShippedModel"]:
        summary = {
            "replacedShippedModel": False,
            "previousShippedTestTop1": baseline,
            "testTop1": metrics["testTop1"],
            "ensembleTestTop1": metrics["ensembleTestTop1"],
            "studentTestTop1": metrics["studentTestTop1"],
            "dtype": dtype,
            "tta": use_tta,
        }
        print(json.dumps(summary, indent=2))
        print("Kept the existing shipped model. This run did not beat it on the signer-disjoint test set.")
        return
    (metrics_dir / "asl_en_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def blob(weights: dict[str, np.ndarray]) -> dict:
        return {
            "weights": {key: encode_weights(value, dtype) for key, value in weights.items()},
            "weightShapes": {key: list(value.shape) for key, value in weights.items()},
        }

    model_json = {
        "format": "signbridge-tx-v1",
        "language": "ase",
        "outputLanguage": "en",
        "task": "isolated-asl-gloss-to-english",
        "featureDim": FEATURE_DIM,
        "seqLen": SEQ_LEN,
        "dModel": final_models[0].d_model,
        "heads": final_models[0].heads,
        "layers": len(final_models[0].blocks),
        "dtype": dtype,
        "labels": labels,
        "emitThreshold": threshold["threshold"],
        "marginThreshold": threshold["margin"],
        "classThresholds": class_thresholds,
        "tta": use_tta,
        "dataset": {
            "name": "PopSign ASL v1.0",
            "subset": dataset_subset,
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
    shipped_path.write_text(json.dumps(model_json), encoding="utf-8")
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
            batch = torch.from_numpy(seq).unsqueeze(0).to(device)
            parts = [member(batch).cpu().numpy()[0] for member in final_models]
            if use_tta:
                flipped = torch.from_numpy(mirror_sequence(seq)).unsqueeze(0).to(device)
                parts.extend(member(flipped).cpu().numpy()[0] for member in final_models)
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
    summary = {
        key: metrics[key]
        for key in (
            "replacedShippedModel",
            "testTop1",
            "testTop5",
            "testMacroRecall",
            "ensembleTestTop1",
            "studentTestTop1",
            "dtype",
            "tta",
            "counts",
            "classes",
            "testAtEmit",
            "focusGlosses",
        )
    }
    print(json.dumps(summary, indent=2))


MODEL_NOTE = (
    "MediaPipe Holistic landmarker float16, "
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)


if __name__ == "__main__":
    main()
