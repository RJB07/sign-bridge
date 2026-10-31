#!/usr/bin/env python3
"""Blend the shipped fine members with the signer192 members on validation.

Each family sees its own features (shoulder-scaled shape vs palm-scaled shape).
Logits are averaged with weights chosen on the validation signers. The test
split is not read. A later --report-test scores it once, and only when this
file's saved validation top-1 is above the shipped 55.5%.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from sign_transformer import SignTransformer
from train_sequence_model import SequenceDataset, evaluate, load_index, ranked_accuracy, round_float16

SHIPPED_VAL = 0.5553
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path("/tmp/signbridge-blend")


def load_member(path: Path, n_classes: int, ff: int) -> SignTransformer:
    model = SignTransformer(389, n_classes, d_model=192, heads=4, layers=3, ff=ff, dropout=0.0, shape_embed=True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return round_float16(model)


def val_rows(cache: Path, index: str) -> tuple[list[dict], list[str]]:
    rows = load_index(cache / index)
    labels = sorted({row["label"] for row in rows if row["split"] == "train"})
    label_to_index = {label: i for i, label in enumerate(labels)}
    validation = [row for row in rows if row["split"] == "validation" and row["label"] in label_to_index]
    return validation, labels


def member_logits(model: SignTransformer, rows: list[dict], label_to_index: dict[str, int], seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(
        SequenceDataset(rows, label_to_index, seq_dir, False, 0, preload=True, layout="fine"),
        batch_size=256,
    )
    return evaluate(model, loader, torch.device("cpu"), tta=False)


def cache_family(name: str, directory: Path, rows: list[dict], labels: list[str], seq_dir: Path) -> tuple[list[np.ndarray], np.ndarray]:
    label_to_index = {label: i for i, label in enumerate(labels)}
    packs = []
    target = None
    for seed in (0, 1, 2):
        path = CACHE_DIR / f"{name}-{seed}.npy"
        target_path = CACHE_DIR / f"{name}-{seed}-y.npy"
        if path.exists() and target_path.exists():
            logits = np.load(path)
            y = np.load(target_path)
        else:
            model = load_member(directory / f"member-{seed}.pt", len(labels), 480)
            logits, y = member_logits(model, rows, label_to_index, seq_dir)
            np.save(path, logits.astype(np.float16))
            np.save(target_path, y.astype(np.int16))
            logits = logits.astype(np.float16)
            del model
        packs.append(logits.astype(np.float32))
        if target is None:
            target = y.astype(np.int64)
        elif not np.array_equal(target, y):
            raise SystemExit(f"{name} seed {seed} targets do not match")
        print({"cached": f"{name}-{seed}", "top1": round(ranked_accuracy(packs[-1], target, 1), 4)}, flush=True)
    return packs, target


def combine(packs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    total = float(sum(weights))
    acc = np.zeros_like(packs[0], dtype=np.float32)
    for pack, weight in zip(packs, weights):
        acc += pack * (weight / total)
    return acc


def weight_grid(count: int) -> list[tuple[float, ...]]:
    if count == 1:
        return [(1.0,)]
    if count == 2:
        return [(float(a), float(10 - a)) for a in range(1, 10)]
    steps = (0.5, 1.0, 1.5, 2.0)
    grids = []
    for combo in itertools.product(steps, repeat=count):
        grids.append(tuple(float(value) for value in combo))
    return grids


def search(catalog: list[tuple[str, np.ndarray]], target: np.ndarray, limit: int) -> list[dict]:
    found = []
    for size in range(1, limit + 1):
        for choice in itertools.combinations(range(len(catalog)), size):
            names = [catalog[index][0] for index in choice]
            packs = [catalog[index][1] for index in choice]
            best = None
            for weights in weight_grid(size):
                logits = combine(packs, list(weights))
                top1 = ranked_accuracy(logits, target, 1)
                top5 = ranked_accuracy(logits, target, 5)
                if best is None or top1 > best["valTop1"]:
                    best = {
                        "members": names,
                        "weights": [round(weight / sum(weights), 4) for weight in weights],
                        "valTop1": round(top1, 4),
                        "valTop5": round(top5, 4),
                    }
            found.append(best)
            print({"blend": best}, flush=True)
    found.sort(key=lambda row: (row["valTop1"], row["valTop5"]), reverse=True)
    return found


def prune_ff(model: SignTransformer, keep: int) -> SignTransformer:
    student = SignTransformer(389, model.fc.out_features, d_model=192, heads=4, layers=3, ff=keep, dropout=0.0, shape_embed=True)
    source = model.state_dict()
    dest = student.state_dict()
    for key, value in source.items():
        if key.endswith("ff1.weight"):
            score = value.detach().float().norm(dim=1)
            index = torch.sort(torch.topk(score, keep).indices).values
            dest[key].copy_(value[index])
            dest[key.replace("ff1.weight", "ff1.bias")].copy_(source[key.replace("ff1.weight", "ff1.bias")][index])
            dest[key.replace("ff1.weight", "ff2.weight")].copy_(source[key.replace("ff1.weight", "ff2.weight")][:, index])
        elif key.endswith("ff1.bias") or key.endswith("ff2.weight"):
            continue
        else:
            dest[key].copy_(value)
    student.load_state_dict(dest)
    student.eval()
    return student


def quantize_linear_(model: SignTransformer) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                quantized = torch.round(weight / scale).clamp(-127, 127)
                module.weight.copy_(quantized * scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="training/cache")
    parser.add_argument("--index", default="index.jsonl")
    parser.add_argument("--max-members", type=int, default=4)
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cache = Path(args.cache)
    rows, labels = val_rows(cache, args.index)
    label_to_index = {label: i for i, label in enumerate(labels)}
    fine_packs, fine_y = cache_family("fine", ROOT / "training/artifacts/fine192", rows, labels, cache / "seq-fine")
    signer_packs, signer_y = cache_family("signer", ROOT / "training/artifacts/signer192", rows, labels, cache / "seq-signer")
    if not np.array_equal(fine_y, signer_y):
        raise SystemExit("Fine and signer validation targets differ")
    catalog = [(f"fine-{seed}", pack) for seed, pack in enumerate(fine_packs)]
    catalog += [(f"signer-{seed}", pack) for seed, pack in enumerate(signer_packs)]
    ranked = search(catalog, fine_y, args.max_members)
    fine_only = combine(fine_packs, [1, 1, 1])
    signer_only = combine(signer_packs, [1, 1, 1])
    summary = {
        "shippedValTop1": SHIPPED_VAL,
        "fineEnsembleValTop1": round(ranked_accuracy(fine_only, fine_y, 1), 4),
        "fineEnsembleValTop5": round(ranked_accuracy(fine_only, fine_y, 5), 4),
        "signerEnsembleValTop1": round(ranked_accuracy(signer_only, fine_y, 1), 4),
        "signerEnsembleValTop5": round(ranked_accuracy(signer_only, fine_y, 5), 4),
        "best": ranked[0],
        "top": ranked[:12],
        "testReported": False,
        "counts": {"validation": len(rows)},
    }
    # One int8 pass and one structured prune, on the strongest fine member, to see if either is usable.
    probe = load_member(ROOT / "training/artifacts/fine192/member-2.pt", len(labels), 480)
    base_logits, base_y = member_logits(probe, rows, label_to_index, cache / "seq-fine")
    quantize_linear_(probe)
    q_logits, _ = member_logits(probe, rows, label_to_index, cache / "seq-fine")
    pruned = prune_ff(load_member(ROOT / "training/artifacts/fine192/member-2.pt", len(labels), 480), 240)
    p_logits, _ = member_logits(pruned, rows, label_to_index, cache / "seq-fine")
    summary["latencyProbes"] = {
        "note": "Int8 is dequantized before the same float matmul, so it does not shorten the browser loop. Structured FF pruning does.",
        "fine2ValTop1": round(ranked_accuracy(base_logits, base_y, 1), 4),
        "fine2Int8ValTop1": round(ranked_accuracy(q_logits, base_y, 1), 4),
        "fine2PrunedFf240ValTop1": round(ranked_accuracy(p_logits, base_y, 1), 4),
    }
    out = ROOT / "training/metrics/blend-val.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "best": summary["best"], "fine": summary["fineEnsembleValTop1"], "probes": summary["latencyProbes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
