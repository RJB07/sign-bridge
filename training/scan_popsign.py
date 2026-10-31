#!/usr/bin/env python3
"""Index PopSign frame shards without downloading the images.

Hugging Face serves sign/popsign-images (CC BY 4.0) without a login. Each
parquet is hundreds of megabytes because it stores 256px frames. The gloss
and the signer id live in the small `text` and `file` columns, so this script
range-reads those column chunks plus the parquet footer and writes
training/data/popsign/catalog.jsonl. Landmark extraction then downloads only
the row groups it needs.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from splits import signer_from_path

BASE = "https://huggingface.co/datasets/sign/popsign-images/resolve/main"
ROOT = Path(__file__).resolve().parent / "data" / "popsign"
API = "https://huggingface.co/api/datasets/sign/popsign-images"


def list_shards() -> list[tuple[str, str, str]]:
    payload = json.load(urllib.request.urlopen(API, timeout=60))
    shards = []
    for sibling in payload["siblings"]:
        name = sibling["rfilename"]
        if not name.endswith(".parquet") or "of-00001" in name:
            continue
        subset = name.split("/")[1]
        split = name.split("/")[-1].split("-")[0]
        if subset not in {"game", "non-game"} or split not in {"train", "validation", "test"}:
            continue
        shards.append((subset, split, name))
    return shards


def _open(request, timeout: int):
    import time

    last = None
    for attempt in range(6):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - DNS and CDN blips
            last = exc
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(last)


def _fetch(url: str, start: int, end: int) -> bytes:
    last = None
    for _ in range(4):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            with _open(request, 90) as response:
                data = response.read()
            if len(data) != end - start:
                raise IOError(f"short read {len(data)} != {end - start}")
            return data
        except Exception as exc:  # noqa: BLE001 - retry transient HTTP errors
            last = exc
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def _length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with _open(request, 60) as response:
        return int(response.headers["Content-Length"])


def index_shard(subset: str, split: str, remote_name: str) -> list[dict]:
    url = f"{BASE}/{remote_name}"
    length = _length(url)
    tail = _fetch(url, length - 8, length)
    if tail[-4:] != b"PAR1":
        raise RuntimeError(f"{remote_name} is not parquet")
    footer_len = int.from_bytes(tail[:4], "little")
    footer = _fetch(url, length - footer_len - 8, length)
    local = ROOT / "meta" / remote_name.replace("/", "__")
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("wb") as handle:
        handle.truncate(length)
        handle.seek(length - footer_len - 8)
        handle.write(footer)
    metadata = pq.read_metadata(local)
    ranges = []
    groups = []
    cursor = 0
    for group_index in range(metadata.num_row_groups):
        group = metadata.row_group(group_index)
        groups.append((cursor, group.num_rows))
        cursor += group.num_rows
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            if column.path_in_schema not in {"file", "text"}:
                continue
            offsets = [value for value in (column.dictionary_page_offset, column.data_page_offset) if value is not None]
            start = min(offsets)
            ranges.append((start, start + column.total_compressed_size))
    with local.open("r+b") as handle:
        for start, end in ranges:
            handle.seek(start)
            handle.write(_fetch(url, start, end))
    table = pq.ParquetFile(local).read(columns=["file", "text"])
    files = table.column("file").to_pylist()
    texts = table.column("text").to_pylist()
    rows = []
    parquet_name = remote_name.split("/")[-1]
    for row, (file_name, label) in enumerate(zip(files, texts)):
        group_index = 0
        for index, (start, count) in enumerate(groups):
            if start <= row < start + count:
                group_index = index
                break
        rows.append(
            {
                "subset": subset,
                "split": split,
                "parquet": parquet_name,
                "remote": remote_name,
                "row": row,
                "row_group": group_index,
                "label": str(label),
                "signer": signer_from_path(str(file_name), str(label)),
                "source": str(file_name),
            }
        )
    local.unlink(missing_ok=True)
    return rows


def finished_remotes(path: Path) -> set[str]:
    found = set()
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            found.add(json.loads(line)["remote"])
    return found


def main() -> None:
    shards = list_shards()
    out = ROOT / "catalog.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done_remotes = finished_remotes(out)
    pending = [shard for shard in shards if shard[2] not in done_remotes]
    print(f"indexing {len(pending)} of {len(shards)} shards ({len(done_remotes)} already done)", flush=True)
    done = len(shards) - len(pending)
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(index_shard, subset, split, name): name for subset, split, name in pending}
        for future in as_completed(futures):
            name = futures[future]
            rows = future.result()
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            done += 1
            print(f"[{done}/{len(shards)}] {name} rows={len(rows)}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
