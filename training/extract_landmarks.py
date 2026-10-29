#!/usr/bin/env python3
"""Extract MediaPipe Holistic landmark sequences from PopSign image shards.

Frames are already ~5 fps stills inside each parquet row. Landmarks are not
committed; this writes training/cache only.
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

from features import sequence_features
from splits import (
    assert_no_signer_leakage,
    labels_in_every_split,
    leakage_report,
    select_balanced,
    signer_from_path,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)


def ensure_model(path: Path) -> None:
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading holistic landmarker -> {path}")
    urllib.request.urlretrieve(MODEL_URL, path)


def scan(parquet_dir: Path) -> list[dict]:
    records = []
    for path in sorted(parquet_dir.glob("*.parquet")):
        split = path.name.split("-")[0]
        if split not in {"train", "validation", "test"}:
            continue
        table = pq.ParquetFile(path).read(columns=["file", "text"])
        files = table.column("file").to_pylist()
        texts = table.column("text").to_pylist()
        for row, (file_name, label) in enumerate(zip(files, texts)):
            records.append(
                {
                    "split": split,
                    "label": str(label),
                    "signer": signer_from_path(str(file_name), str(label)),
                    "parquet": path.name,
                    "row": row,
                    "source": str(file_name),
                }
            )
    return records


def make_landmarker(model_path: Path):
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

    options = HolisticLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode_lib.VisionTaskRunningMode.IMAGE,
        min_face_detection_confidence=0.4,
        min_pose_detection_confidence=0.4,
        min_hand_landmarks_confidence=0.4,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    return HolisticLandmarker.create_from_options(options)


def points_or_none(landmarks, expected: int):
    if not landmarks or len(landmarks) < expected:
        return None
    return [[float(p.x), float(p.y), float(p.z)] for p in landmarks[:expected]]


def clip_features(landmarker, image_bytes_list) -> np.ndarray:
    from mediapipe.tasks.python.vision.core import image as image_lib

    frames = []
    for item in image_bytes_list:
        raw = item["bytes"] if isinstance(item, dict) else item
        rgb = np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        mp_image = image_lib.Image(image_lib.ImageFormat.SRGB, rgb)
        result = landmarker.detect(mp_image)
        frames.append(
            {
                "left": points_or_none(result.left_hand_landmarks, 21),
                "right": points_or_none(result.right_hand_landmarks, 21),
                "pose": points_or_none(result.pose_landmarks, 33),
            }
        )
    return sequence_features(frames)


def extract(parquet_dir: Path, selected: list[dict], out_dir: Path, model_path: Path, prefix: str = "") -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = out_dir / "seq"
    seq_dir.mkdir(exist_ok=True)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for record in selected:
        by_file[record["parquet"]].append(record)

    landmarker = make_landmarker(model_path)
    written = []
    try:
        for name, rows in sorted(by_file.items()):
            path = parquet_dir / name
            print(f"[{prefix or 'main'}] extract {name} ({len(rows)} clips)", flush=True)
            table = pq.ParquetFile(path).read(columns=["images"])
            images = table.column("images")
            for record in rows:
                feats = clip_features(landmarker, images[record["row"]].as_py())
                if feats.shape[0] < 2:
                    continue
                npy_name = f"{prefix}{record['split']}_{len(written):05d}.npy"
                np.save(seq_dir / npy_name, feats)
                written.append(
                    {
                        "split": record["split"],
                        "label": record["label"],
                        "signer": record["signer"],
                        "npy": npy_name,
                        "frames": int(feats.shape[0]),
                        "source": record["source"],
                    }
                )
                if len(written) % 200 == 0:
                    print(f"[{prefix or 'main'}] saved {len(written)}", flush=True)
            del table
            del images
    finally:
        landmarker.close()
    index_path = out_dir / f"index{prefix}.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in written:
            handle.write(json.dumps(row) + "\n")
    print(f"[{prefix or 'main'}] wrote {len(written)} -> {index_path}", flush=True)
    return written


def _extract_worker(payload: dict) -> str:
    extract(
        Path(payload["parquet_dir"]),
        payload["selected"],
        Path(payload["out"]),
        Path(payload["model"]),
        prefix=payload["prefix"],
    )
    return payload["prefix"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", default="training/data/popsign/full")
    parser.add_argument("--out", default="training/cache")
    parser.add_argument("--model", default="training/models/holistic_landmarker.task")
    parser.add_argument("--max-train", type=int, default=64)
    parser.add_argument("--max-val", type=int, default=20)
    parser.add_argument("--max-test", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    out_dir = Path(args.out)
    model_path = Path(args.model)
    ensure_model(model_path)

    records = scan(parquet_dir)
    assert_no_signer_leakage(records)
    labels = labels_in_every_split(records)
    selected = select_balanced(
        records,
        labels,
        {"train": args.max_train, "validation": args.max_val, "test": args.max_test},
    )
    summary = {
        "labels": labels,
        "selected": len(selected),
        "leakage": leakage_report(records),
        "counts": {
            split: sum(1 for row in selected if row["split"] == split)
            for split in ("train", "validation", "test")
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"labels": len(labels), "counts": summary["counts"], "leakage": summary["leakage"]}, indent=2))
    files = sorted({row["parquet"] for row in selected})
    workers = max(1, args.workers)
    if workers == 1:
        extract(parquet_dir, selected, out_dir, model_path)
        return
    import multiprocessing as mp

    groups = [files[i::workers] for i in range(workers)]
    ctx = mp.get_context("spawn")
    processes = []
    for index, group in enumerate(groups):
        group_set = set(group)
        payload = {
            "parquet_dir": str(parquet_dir),
            "selected": [row for row in selected if row["parquet"] in group_set],
            "out": str(out_dir),
            "model": str(model_path),
            "prefix": f"w{index}_",
        }
        process = ctx.Process(target=_extract_worker, args=(payload,))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise SystemExit(f"worker failed with {process.exitcode}")
    merged = []
    for index in range(workers):
        part = out_dir / f"indexw{index}_.jsonl"
        if part.exists():
            merged.extend(json.loads(line) for line in part.read_text().splitlines() if line.strip())
    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(merged)} sequences -> {index_path}", flush=True)


if __name__ == "__main__":
    main()
