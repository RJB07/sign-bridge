"""Signer-independent split checks for PopSign clips.

Official PopSign train, validation, and test partitions already assign each
signer to one partition. This module refuses to train if a signer id appears
in more than one partition.
"""

from __future__ import annotations

import random
from collections import defaultdict


def signer_from_path(path: str, gloss: str) -> str:
    base = path.split("/")[-1]
    token = f"-{gloss}-"
    if token in base:
        return base.split(token)[0]
    return base.split("-")[0]


def signer_sets(records: list[dict]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[record["split"]].add(record["signer"])
    return grouped


def leakage_report(records: list[dict]) -> dict[str, list[str]]:
    grouped = signer_sets(records)
    names = ["train", "validation", "test"]
    report = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(grouped.get(left, set()) & grouped.get(right, set()))
            report[f"{left}&{right}"] = overlap
    return report


def assert_no_signer_leakage(records: list[dict]) -> None:
    report = leakage_report(records)
    leaked = {key: value for key, value in report.items() if value}
    if leaked:
        raise SystemExit(f"Signer leakage across splits: {leaked}")


def labels_in_every_split(records: list[dict]) -> list[str]:
    present: dict[str, set[str]] = defaultdict(set)
    for record in records:
        present[record["label"]].add(record["split"])
    needed = {"train", "validation", "test"}
    return sorted(label for label, splits in present.items() if needed <= splits)


def select_balanced(
    records: list[dict],
    labels: list[str],
    caps: dict[str, int],
    seed: int = 0,
) -> list[dict]:
    """Round-robin across signers so one person cannot fill a class."""
    allowed = set(labels)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["label"] in allowed:
            grouped[(record["split"], record["label"])].append(record)

    rng = random.Random(seed)
    selected: list[dict] = []
    for key, rows in grouped.items():
        split = key[0]
        cap = caps.get(split, 0)
        if cap <= 0:
            continue
        by_signer: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_signer[row["signer"]].append(row)
        for signer_rows in by_signer.values():
            rng.shuffle(signer_rows)
        order = list(by_signer)
        rng.shuffle(order)
        picked: list[dict] = []
        cursor = 0
        guard = 0
        while len(picked) < cap and any(by_signer[name] for name in order):
            name = order[cursor % len(order)]
            if by_signer[name]:
                picked.append(by_signer[name].pop())
            cursor += 1
            guard += 1
            if guard > cap * max(1, len(order)) + 5:
                break
        selected.extend(picked)
    return selected
