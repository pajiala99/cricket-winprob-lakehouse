"""Pack Cricsheet-format match zips into newline-delimited JSON shards.

landing/downloads/**/*.zip  ->  landing/packed/<batch_id>/part-00000.ndjson

Why: Spark is slow on tens of thousands of tiny multi-line JSON documents (file listing, per-file open,
no splitting). Packing ~400 matches per shard gives Spark large, splittable inputs.
A manifest (match_id -> CRC32 of the source member) makes the step incremental and idempotent: a re-run
emits only matches that are new or whose content changed (e.g. a re-published revision).
Malformed documents are moved to landing/rejected/ instead of failing the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from cricklake.config import get_settings

log = logging.getLogger("cricklake.pack")


def list_members(zip_path: str) -> list[tuple[str, str, int]]:
    with zipfile.ZipFile(zip_path) as zf:
        return [(zip_path, i.filename, i.CRC) for i in zf.infolist() if i.filename.endswith(".json")]


def write_shard(task) -> tuple[int, list[str]]:
    shard_path, members, rejected_dir = task
    handles: dict[str, zipfile.ZipFile] = {}
    written, rejected = 0, []
    tmp = shard_path + ".tmp"
    with open(tmp, "w") as out:
        for zip_path, member, _crc in members:
            zf = handles.setdefault(zip_path, zipfile.ZipFile(zip_path))
            raw = zf.read(member)
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                Path(rejected_dir).mkdir(parents=True, exist_ok=True)
                (Path(rejected_dir) / Path(member).name).write_bytes(raw)
                rejected.append(member)
                continue
            doc["match_id"] = Path(member).stem
            out.write(json.dumps(doc, separators=(",", ":")))
            out.write("\n")
            written += 1
    os.replace(tmp, shard_path)  # atomic publish: the streaming reader never sees a half-written shard
    for zf in handles.values():
        zf.close()
    return written, rejected


def pack(pattern: str = "**/*.zip", shard_size: int | None = None, workers: int | None = None) -> dict:
    cfg = get_settings()
    downloads = cfg.landing / "downloads"
    packed = cfg.landing / "packed"
    packed.mkdir(parents=True, exist_ok=True)
    manifest_path = packed / "_manifest.json"
    manifest: dict[str, int] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    zips = sorted(str(p) for p in downloads.glob(pattern))
    if not zips:
        raise FileNotFoundError(f"no zip files match {downloads}/{pattern}; run the simulator or download data")
    with ProcessPoolExecutor(workers) as pool:
        listings = list(pool.map(list_members, zips))

    todo = [(z, m, crc) for listing in listings for (z, m, crc) in listing if manifest.get(Path(m).stem) != crc]
    if not todo:
        log.info("Nothing new in %d zip(s); manifest already covers %d matches", len(zips), len(manifest))
        return {"batch_id": None, "matches": 0, "rejected": 0, "shards": 0}

    # timestamp + random suffix: two runs in the same second must not share (and crash on) one batch folder
    batch_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{os.urandom(3).hex()}"
    out_dir = packed / batch_id
    out_dir.mkdir(parents=True)
    size = shard_size or get_settings()["source"]["shard_size"]
    rejected_dir = str(cfg.landing / "rejected" / batch_id)
    tasks = [(str(out_dir / f"part-{i // size:05d}.ndjson"), todo[i:i + size], rejected_dir)
             for i in range(0, len(todo), size)]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(write_shard, tasks))

    for _z, member, crc in todo:
        manifest[Path(member).stem] = crc
    manifest_path.write_text(json.dumps(manifest))
    written = sum(r[0] for r in results)
    rejected = [m for r in results for m in r[1]]
    log.info("Packed %s matches into %d shard(s) (batch %s); rejected %d malformed file(s)",
             f"{written:,}", len(tasks), batch_id, len(rejected))
    return {"batch_id": batch_id, "matches": written, "rejected": len(rejected), "shards": len(tasks)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pattern", default="**/*.zip", help="glob under landing/downloads/")
    ap.add_argument("--shard-size", type=int)
    args = ap.parse_args(argv)
    pack(args.pattern, args.shard_size)


if __name__ == "__main__":
    main()
