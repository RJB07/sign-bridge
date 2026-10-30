"""Choose PopSign clips without letting a test signer into training.

Game clips keep the official partition. Non-game clips are extra training
examples only when that signer is absent from the game validation and test
partitions. The capped game test selection is what the published top-1 uses.
"""

from __future__ import annotations

from splits import assert_no_signer_leakage, labels_in_every_split, leakage_report, select_balanced


def _ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row.get("parquet", ""), int(row.get("row", 0)), row.get("source", "")))


def game_rows(records: list[dict]) -> list[dict]:
    return _ordered([row for row in records if row.get("subset", "game") == "game"])


def select_game(records: list[dict], caps: dict[str, int], seed: int = 0) -> tuple[list[dict], list[str]]:
    rows = game_rows(records)
    assert_no_signer_leakage(rows)
    labels = labels_in_every_split(rows)
    selected = select_balanced(rows, labels, caps, seed=seed)
    for row in selected:
        row["subset"] = "game"
    return selected, labels


def blocked_signers(records: list[dict]) -> set[str]:
    """Signers who must not appear in any added training clip."""
    blocked = set()
    for row in game_rows(records):
        if row["split"] in {"validation", "test"}:
            blocked.add(row["signer"])
    return blocked


def select_nongame_train(
    records: list[dict],
    labels: list[str],
    cap: int,
    seed: int = 0,
) -> tuple[list[dict], dict[str, int]]:
    blocked = blocked_signers(records)
    allowed = set(labels)
    kept = []
    dropped = 0
    for row in records:
        if row.get("subset") != "non-game" or row.get("split") != "train":
            continue
        if row["label"] not in allowed:
            dropped += 1
            continue
        if row["signer"] in blocked:
            dropped += 1
            continue
        kept.append(row)
    picked = select_balanced(_ordered(kept), labels, {"train": cap}, seed=seed)
    for row in picked:
        row["subset"] = "non-game"
        row["split"] = "train"
    return picked, {"candidates": len(kept), "dropped": dropped, "selected": len(picked), "blockedSigners": len(blocked)}


def assert_merged_disjoint(rows: list[dict]) -> dict[str, list[str]]:
    report = leakage_report(rows)
    assert_no_signer_leakage(rows)
    return report


def _split_rows(records: list[dict], split: str) -> list[dict]:
    rows = [row for row in records if row.get("subset", "game") == "game" and row["split"] == split]
    return _ordered(rows)


def choose(
    records: list[dict],
    train_cap: int,
    val_cap: int,
    test_cap: int,
    nongame_cap: int,
    core_cap: int = 64,
) -> tuple[list[dict], dict]:
    """Each split is drawn on its own so the test clips do not depend on the train cap.

    Rows are sorted before the draw, so a catalog written in download order
    selects the same clips on the next machine. `core` is a core_cap-per-gloss
    subset of the extracted training clips. It is not the published 2,744-clip
    list: that run used a partial shard download.
    """
    game = [row for row in records if row.get("subset", "game") == "game"]
    labels = labels_in_every_split(game)
    validation = select_balanced(_split_rows(records, "validation"), labels, {"validation": val_cap}, seed=0)
    test = select_balanced(_split_rows(records, "test"), labels, {"test": test_cap}, seed=0)
    train = select_balanced(_split_rows(records, "train"), labels, {"train": train_cap}, seed=0)
    core_rows = select_balanced(train, labels, {"train": core_cap}, seed=1)
    core_ids = {(row["parquet"], int(row["row"])) for row in core_rows}
    game_selected = validation + test + train
    for row in game_selected:
        row["subset"] = "game"
        row["core"] = row["split"] == "train" and (row["parquet"], int(row["row"])) in core_ids
    extra, extra_stats = select_nongame_train(records, labels, nongame_cap, seed=0)
    for row in extra:
        row["core"] = False
    merged = game_selected + extra
    leakage = assert_merged_disjoint(merged)
    note = {
        "license": "CC BY 4.0",
        "source": "https://huggingface.co/datasets/sign/popsign-images",
        "subsets": ["game", "non-game"],
        "labels": len(labels),
        "caps": {
            "train": train_cap,
            "validation": val_cap,
            "test": test_cap,
            "nongameTrain": nongame_cap,
            "coreTrain": core_cap,
        },
        "gameCounts": {
            "train": sum(1 for row in game_selected if row["split"] == "train"),
            "validation": sum(1 for row in game_selected if row["split"] == "validation"),
            "test": sum(1 for row in game_selected if row["split"] == "test"),
            "coreTrain": sum(1 for row in game_selected if row["split"] == "train" and row.get("core")),
        },
        "nongameTrain": extra_stats,
        "leakage": leakage,
        "testProtocol": (
            "Official PopSign game partition, seed 0, at most test_cap clips per gloss, "
            "clips ordered by parquet name and row before the draw. "
            "The 50.6% model used this selector on a partial shard download, so its clip list was smaller."
        ),
    }
    return merged, note
