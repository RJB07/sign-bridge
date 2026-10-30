#!/usr/bin/env python3
"""Download only the PopSign row groups we train on and cache landmarks.

The frame parquets are about 90 GB. `scan_popsign.py` already indexed gloss
and signer for every clip. This script keeps the official game split, adds
non-game training clips whose signers are not in game validation or test, and
range-downloads the image bytes for the selected row groups.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from features import LIP_INDEXES, sequence_features
from sources import choose, clips_to_extract

BASE = "https://huggingface.co/datasets/sign/popsign-images/resolve/main"
ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "public" / "models" / "holistic_landmarker.task"
WANTED = {"file", "text", "images.list.element.bytes", "images.list.element.path"}


def load_catalog(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _fetch(url: str, start: int, end: int) -> bytes:
    last = None
    for _ in range(5):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            if len(data) != end - start:
                raise IOError(f"short read {len(data)} != {end - start}")
            return data
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"{url} {start}-{end} failed: {last}")


def _length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def materialize(remote: str, row_groups: set[int], dest: Path) -> list[int]:
    url = f"{BASE}/{remote}"
    length = _length(url)
    tail = _fetch(url, length - 8, length)
    if tail[-4:] != b"PAR1":
        raise RuntimeError(f"{remote} is not parquet")
    footer_len = int.from_bytes(tail[:4], "little")
    footer = _fetch(url, length - footer_len - 8, length)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        handle.truncate(length)
        handle.seek(length - footer_len - 8)
        handle.write(footer)
    metadata = pq.read_metadata(dest)
    starts = []
    cursor = 0
    with dest.open("r+b") as handle:
        for group_index in range(metadata.num_row_groups):
            group = metadata.row_group(group_index)
            starts.append(cursor)
            cursor += group.num_rows
            if group_index not in row_groups:
                continue
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                if column.path_in_schema not in WANTED:
                    continue
                offsets = [value for value in (column.dictionary_page_offset, column.data_page_offset) if value is not None]
                start = min(offsets)
                handle.seek(start)
                handle.write(_fetch(url, start, start + column.total_compressed_size))
    return starts


def make_landmarker(model_path: Path):
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

    options = HolisticLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode_lib.VisionTaskRunningMode.IMAGE,
        min_face_detection_confidence=0.4,
        min_pose_detection_confidence=0.4,
        min_hand_landmarks_confidence=0.35,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    return HolisticLandmarker.create_from_options(options)


def _points(landmarks, expected: int):
    if not landmarks or len(landmarks) < expected:
        return None
    return [[float(point.x), float(point.y), float(point.z)] for point in landmarks[:expected]]


def _lips(landmarks) -> list[list[float]] | None:
    if not landmarks or len(landmarks) <= max(LIP_INDEXES):
        return None
    return [[float(landmarks[index].x), float(landmarks[index].y), float(landmarks[index].z)] for index in LIP_INDEXES]


def _pack_raw(frames: list[dict]) -> np.ndarray:
    width = 21 + 21 + 33 + len(LIP_INDEXES)
    raw = np.full((len(frames), width, 3), np.nan, dtype=np.float32)
    for time, frame in enumerate(frames):
        cursor = 0
        for key, count in (("left", 21), ("right", 21), ("pose", 33), ("lips", len(LIP_INDEXES))):
            points = frame.get(key)
            if points is not None:
                raw[time, cursor : cursor + count] = np.asarray(points, dtype=np.float32)[:count]
            cursor += count
    return raw


def clip_arrays(landmarker, image_bytes_list):
    from mediapipe.tasks.python.vision.core import image as image_lib

    frames = []
    for item in image_bytes_list:
        raw = item["bytes"] if isinstance(item, dict) else item
        rgb = np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        mp_image = image_lib.Image(image_lib.ImageFormat.SRGB, rgb)
        result = landmarker.detect(mp_image)
        frames.append(
            {
                "left": _points(result.left_hand_landmarks, 21),
                "right": _points(result.right_hand_landmarks, 21),
                "pose": _points(result.pose_landmarks, 33),
                "lips": _lips(result.face_landmarks),
            }
        )
    features = sequence_features(frames)
    return features, _pack_raw(frames)


def npy_name(record: dict) -> str:
    shard = record["parquet"].replace(".parquet", "")
    subset = "game" if record.get("subset", "game") == "game" else "nongame"
    return f"{subset}_{record['split']}_{shard}_{int(record['row']):05d}.npy"


def extract_remote(payload: dict) -> str:
    remote = payload["remote"]
    records = payload["records"]
    out_dir = Path(payload["out"])
    seq_dir = out_dir / "seq"
    raw_dir = out_dir / "raw"
    meta_dir = out_dir / "meta"
    seq_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for record in records:
        name = npy_name(record)
        meta_path = meta_dir / f"{name}.json"
        if (seq_dir / name).exists() and meta_path.exists():
            continue
        pending.append(record)
    if not pending:
        return f"skip {remote}"
    groups = {int(record["row_group"]) for record in pending}
    sparse = Path(payload["tmp"]) / remote.replace("/", "__")
    starts = materialize(remote, groups, sparse)
    landmarker = make_landmarker(Path(payload["model"]))
    written = 0
    try:
        parquet = pq.ParquetFile(sparse)
        by_group = defaultdict(list)
        for record in pending:
            by_group[int(record["row_group"])].append(record)
        for group_index, group_rows in by_group.items():
            table = parquet.read_row_group(group_index, columns=["images"])
            images = table.column("images")
            start = starts[group_index]
            for record in group_rows:
                local = int(record["row"]) - start
                try:
                    features, raw = clip_arrays(landmarker, images[local].as_py())
                except Exception as exc:  # noqa: BLE001 - one bad frame should not drop the shard
                    print(f"clip fail {record.get('source')}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if features.shape[0] < 2:
                    continue
                name = npy_name(record)
                np.save(seq_dir / name, features)
                np.save(raw_dir / name, raw.astype(np.float16))
                meta = {
                    "split": record["split"],
                    "label": record["label"],
                    "signer": record["signer"],
                    "npy": name,
                    "frames": int(features.shape[0]),
                    "source": record["source"],
                    "subset": record.get("subset", "game"),
                    "parquet": record["parquet"],
                    "row": int(record["row"]),
                    "core": bool(record.get("core", False)),
                }
                (meta_dir / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
                written += 1
            del table
            del images
    finally:
        landmarker.close()
        sparse.unlink(missing_ok=True)
    return f"wrote {written} from {remote} groups={sorted(groups)}"


def _worker(payload: dict) -> str:
    import os
    import time

    os.environ["GLOG_minloglevel"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    last = "unknown"
    for attempt in range(3):
        try:
            return extract_remote(payload)
        except Exception as exc:  # noqa: BLE001 - CDN blips should not drop the shard
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(20, 2 ** attempt))
    return f"ERR {payload['remote']} {last}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="training/data/popsign/catalog.jsonl")
    parser.add_argument("--out", default="training/cache")
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--train-cap", type=int, default=160)
    parser.add_argument("--val-cap", type=int, default=20)
    parser.add_argument("--test-cap", type=int, default=20)
    parser.add_argument("--nongame-cap", type=int, default=80)
    parser.add_argument("--core-cap", type=int, default=64)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Download validation, test, and the core training subset. The core ids are still drawn from the train cap.",
    )
    parser.add_argument(
        "--with-nongame",
        action="store_true",
        help="With --core-only, also download non-game training clips. Game validation and test signers stay out.",
    )
    parser.add_argument("--progress", default="", help="Append one JSON status line per finished shard.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_catalog(Path(args.catalog))
    selected, note = choose(
        records,
        args.train_cap,
        args.val_cap,
        args.test_cap,
        args.nongame_cap,
        core_cap=args.core_cap,
    )
    selected = clips_to_extract(selected, core_only=args.core_only, with_nongame=args.with_nongame)
    if args.core_only and args.with_nongame:
        note["extracted"] = "validation, test, core train, and non-game train"
    elif args.core_only:
        note["extracted"] = "validation, test, and core train"
    print(json.dumps({key: note[key] for key in ("labels", "caps", "gameCounts", "nongameTrain", "leakage") if key in note}, indent=2))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    if args.dry_run:
        groups = defaultdict(set)
        for row in selected:
            groups[row["remote"]].add(int(row["row_group"]))
        print(json.dumps({"clips": len(selected), "shards": len(groups), "rowGroups": sum(len(v) for v in groups.values())}))
        return

    by_remote = defaultdict(list)
    for row in selected:
        by_remote[row["remote"]].append(row)
    tmp = Path("/tmp/popsign-shards")
    tmp.mkdir(parents=True, exist_ok=True)
    payloads = [
        {"remote": remote, "records": rows, "out": str(out_dir), "model": args.model, "tmp": str(tmp)}
        for remote, rows in sorted(by_remote.items())
    ]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    done = 0
    progress_path = Path(args.progress) if args.progress else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    with ctx.Pool(processes=max(1, args.workers)) as pool:
        for message in pool.imap_unordered(_worker, payloads):
            done += 1
            print(f"[{done}/{len(payloads)}] {message}", flush=True)
            if progress_path is not None:
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"done": done, "total": len(payloads), "message": message}) + "\n")
            if message.startswith("ERR"):
                raise SystemExit(message)
    metas = sorted((out_dir / "meta").glob("*.json"))
    index_path = out_dir / "index.jsonl"
    core_path = out_dir / "index-core.jsonl"
    with index_path.open("w", encoding="utf-8") as handle, core_path.open("w", encoding="utf-8") as core_handle:
        for path in metas:
            line = path.read_text(encoding="utf-8").strip()
            handle.write(line + "\n")
            row = json.loads(line)
            if row["split"] != "train" or row.get("core"):
                core_handle.write(line + "\n")
    print(f"wrote {len(metas)} -> {index_path}", flush=True)


if __name__ == "__main__":
    main()
