import json
import zipfile

from cricklake.ingest import pack as pack_mod
from tests.conftest import make_settings, simulated_matches


def _write_zip(path, docs, corrupt=False):
    with zipfile.ZipFile(path, "w") as zf:
        for d in docs:
            zf.writestr(f"{d['match_id']}.json", json.dumps({k: v for k, v in d.items() if k != "match_id"}))
        if corrupt:
            zf.writestr("999.json", '{"info": {"dates": ["20')


def test_pack_is_incremental_idempotent_and_rejects_bad_files(tmp_path, monkeypatch):
    cfg = make_settings(tmp_path)
    monkeypatch.setattr(pack_mod, "get_settings", lambda: cfg)
    downloads = cfg.landing / "downloads"
    downloads.mkdir(parents=True)
    docs = simulated_matches(n=3)
    _write_zip(downloads / "drop1.zip", docs, corrupt=True)

    first = pack_mod.pack("*.zip", shard_size=2, workers=1)
    assert first["matches"] == 3 and first["rejected"] == 1 and first["shards"] == 2

    shard_lines = [json.loads(line) for f in (cfg.landing / "packed").rglob("*.ndjson") for line in f.open()]
    assert sorted(d["match_id"] for d in shard_lines) == ["9000", "9001", "9002"]

    assert pack_mod.pack("*.zip", workers=1)["matches"] == 0  # nothing new -> no-op

    docs[0]["meta"]["revision"] = 2                            # a re-published match
    _write_zip(downloads / "drop1.zip", docs)
    assert pack_mod.pack("*.zip", workers=1)["matches"] == 1
