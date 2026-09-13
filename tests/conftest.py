import itertools
import json
import os
import random
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from cricklake.config import CONFIG_FILE, Settings
from cricklake.ingest import simulator as sim


def make_settings(root: Path) -> Settings:
    """Project config with every path redirected into a temporary directory."""
    raw = yaml.safe_load(open(CONFIG_FILE))
    raw["paths"] = {key: str(root / key) for key in raw["paths"]}
    return Settings(raw)


def simulated_matches(n: int = 6, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    used, serial = set(), itertools.count()
    teams = []
    for k, city in enumerate(["Patiala", "Shimla"]):
        venue = sim.Venue(f"{city} Stadium", city, 0.0, 0.01)
        squad = [sim.new_player(rng, 0, next(serial), role, "male", used) for role in sim.SQUAD_ROLES]
        teams.append(sim.Team(f"{city} Testers", f"team_{k}", city, venue, squad))
    docs = []
    for i in range(n):
        match, *_ = sim.simulate_match(rng, "Test League", "male", "2024", 2024,
                                       {"name": "Test League", "match_number": i + 1},
                                       teams[i % 2], teams[(i + 1) % 2], date(2024, 4, 1) + timedelta(days=i), 0.0)
        match["match_id"] = str(9000 + i)
        docs.append(match)
    return docs


def bronze_frame(spark, docs: list[dict], path: Path):
    from pyspark.sql import functions as F

    from cricklake.schemas import MATCH

    path.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    return (spark.read.schema(MATCH).option("columnNameOfCorruptRecord", "_corrupt_record").json(str(path))
            .withColumn("_source_file", F.lit(str(path)))
            .withColumn("_batch_id", F.lit(path.stem))
            .withColumn("_ingested_at", F.current_timestamp()))


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("SPARK_MASTER", "local[2]")
    from cricklake.spark import get_spark

    session = get_spark("tests")
    session.conf.set("spark.sql.shuffle.partitions", "4")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def lakehouse(spark, tmp_path_factory):
    """A tiny silver layer built from six simulated matches."""
    from cricklake.pipelines import silver

    root = tmp_path_factory.mktemp("lake")
    cfg = make_settings(root)
    docs = simulated_matches()
    silver.process_batch(bronze_frame(spark, docs, root / "batch0.ndjson"), 0, cfg, aliases=None)
    return cfg, docs
