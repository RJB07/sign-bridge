"""Non-game clips cannot move a held-out signer into training."""

import unittest

from sources import assert_merged_disjoint, choose, clips_to_extract, select_game, select_nongame_train


def row(subset, split, signer, label, parquet="a.parquet", index=0):
    return {
        "subset": subset,
        "split": split,
        "signer": signer,
        "label": label,
        "parquet": parquet,
        "row": index,
        "source": f"{split}/{label}/{signer}-{label}-x.mp4",
    }


class SourceTests(unittest.TestCase):
    def test_nongame_drops_test_and_validation_signers(self):
        records = [
            row("game", "train", "s-train", "apple", index=0),
            row("game", "validation", "s-val", "apple", index=1),
            row("game", "test", "s-test", "apple", index=2),
            row("non-game", "train", "s-train", "apple", "n.parquet", 0),
            row("non-game", "train", "s-new", "apple", "n.parquet", 1),
            row("non-game", "train", "s-test", "apple", "n.parquet", 2),
            row("non-game", "train", "s-val", "apple", "n.parquet", 3),
            row("non-game", "test", "s-new", "apple", "n.parquet", 4),
        ]
        selected, labels = select_game(records, {"train": 5, "validation": 5, "test": 5}, seed=0)
        self.assertEqual(labels, ["apple"])
        extra, stats = select_nongame_train(records, labels, cap=5, seed=0)
        signers = {item["signer"] for item in extra}
        self.assertEqual(signers, {"s-train", "s-new"})
        self.assertGreaterEqual(stats["dropped"], 2)
        merged = selected + extra
        report = assert_merged_disjoint(merged)
        self.assertEqual(report["train&test"], [])
        self.assertEqual(report["validation&test"], [])
        self.assertEqual(report["train&validation"], [])

    def test_choose_is_stable_when_catalog_order_changes(self):
        records = []
        index = 0
        for label in ("apple", "bird"):
            for split, signer in (("train", "s-train"), ("validation", "s-val"), ("test", "s-test")):
                for copy in range(4):
                    records.append(row("game", split, signer, label, f"{split}.parquet", index))
                    index += 1
            records.append(row("non-game", "train", "s-test", label, "n.parquet", index))
            index += 1
            records.append(row("non-game", "train", "s-new", label, "n.parquet", index))
            index += 1
        first, note = choose(records, train_cap=3, val_cap=2, test_cap=2, nongame_cap=1, core_cap=2)
        shuffled = list(reversed(records))
        second, _note = choose(shuffled, train_cap=3, val_cap=2, test_cap=2, nongame_cap=1, core_cap=2)
        key = lambda item: (item["subset"], item["split"], item["parquet"], item["row"], item["core"])
        self.assertEqual(sorted(map(key, first)), sorted(map(key, second)))
        self.assertEqual(note["leakage"]["train&test"], [])
        core = [item for item in first if item["split"] == "train" and item.get("core")]
        self.assertLessEqual(len(core), 4)
        self.assertTrue(core)
        self.assertTrue(all(item["subset"] == "game" for item in core))

    def test_core_extract_can_keep_nongame_training_clips(self):
        records = []
        index = 0
        for label in ("apple", "bird"):
            for split, signer in (("train", "s-train"), ("validation", "s-val"), ("test", "s-test")):
                for _copy in range(4):
                    records.append(row("game", split, signer, label, f"{split}.parquet", index))
                    index += 1
            records.append(row("non-game", "train", "s-new", label, "n.parquet", index))
            index += 1
        selected, _note = choose(records, train_cap=3, val_cap=2, test_cap=2, nongame_cap=5, core_cap=2)
        core = clips_to_extract(selected, core_only=True, with_nongame=False)
        both = clips_to_extract(selected, core_only=True, with_nongame=True)
        self.assertTrue(core)
        self.assertTrue(all(item["split"] != "train" or item.get("core") for item in core))
        self.assertFalse(any(item["subset"] == "non-game" for item in core))
        nongame = [item for item in both if item["subset"] == "non-game"]
        self.assertEqual(len(nongame), 2)
        self.assertTrue(all(item["split"] == "train" and item["signer"] == "s-new" for item in nongame))
        heldout = {(item["split"], item["parquet"], item["row"]) for item in both if item["split"] != "train"}
        self.assertEqual(heldout, {(item["split"], item["parquet"], item["row"]) for item in core if item["split"] != "train"})


if __name__ == "__main__":
    unittest.main()
