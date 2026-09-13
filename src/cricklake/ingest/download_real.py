"""Fetch real Cricsheet match files and package them like cricsheet.org's per-competition zips.

cricsheet.org sits behind an anti-bot page, so the files come from a public Hugging Face mirror of its
downloads. Each competition becomes landing/downloads/cricsheet/<competition>.zip with one <match_id>.json
member per match: exactly the layout of the official zips, so the pack step needs no changes.
Data: Cricsheet (https://cricsheet.org), attribution required.

  CRICKLAKE_CONFIG=conf/real.yaml python -m cricklake.ingest.download_real
"""

from __future__ import annotations

import argparse
import logging
import zipfile

from cricklake.config import get_settings

log = logging.getLogger("cricklake.download_real")


def main(argv=None) -> dict:
    cfg = get_settings()
    source = cfg["source"]
    ap = argparse.ArgumentParser(description="Download real Cricsheet data from the Hugging Face mirror")
    ap.add_argument("--competitions", nargs="+", default=source.get("competitions", ["ipl_json"]))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    from huggingface_hub import snapshot_download

    mirror = cfg.landing.parent / "hf_mirror"
    snapshot_download(source["mirror_repo"], repo_type="dataset", local_dir=str(mirror),
                      allow_patterns=[f"{c}/*.json" for c in args.competitions], max_workers=args.workers)

    out_dir = cfg.landing / "downloads" / "cricsheet"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for competition in args.competitions:
        files = sorted((mirror / competition).glob("*.json"))
        target = out_dir / f"{competition}.zip"
        tmp = target.with_name(target.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        tmp.replace(target)
        counts[competition] = len(files)
        log.info("%-10s %5d matches -> %s", competition, len(files), target)
    log.info("Real data ready: %s matches in %d zips", f"{sum(counts.values()):,}", len(counts))
    return counts


if __name__ == "__main__":
    main()
