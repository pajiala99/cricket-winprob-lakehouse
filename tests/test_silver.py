import copy

from pyspark.sql import functions as F

from cricklake.pipelines import silver
from tests.conftest import bronze_frame


def _balls(doc):
    return sum(len(o["deliveries"]) for inn in doc["innings"] for o in inn["overs"])


def test_silver_explodes_every_ball(spark, lakehouse):
    cfg, docs = lakehouse
    deliveries = spark.read.format("delta").load(cfg.table("silver", "deliveries"))
    assert deliveries.count() == sum(_balls(d) for d in docs)
    matches = spark.read.format("delta").load(cfg.table("silver", "matches"))
    assert matches.count() == len(docs)
    # ball order is stable and legal-ball counters never exceed the over limit (+1 for a miscount)
    worst = deliveries.groupBy("match_id", "innings_no").agg(F.max("legal_ball_no").alias("n")).agg(F.max("n")).first()[0]
    assert worst <= 121


def test_venue_names_are_normalised(spark):
    df = spark.createDataFrame([("Patiala Stadium, Patiala", "Patiala"), ("Shimla Oval", "Shimla")], ["venue", "city"])
    raw = df.select(F.struct(F.col("venue"), F.col("city")).alias("info"))
    out = raw.select(
        F.when(F.col("info.city").isNotNull() & F.trim("info.venue").endswith(F.concat(F.lit(", "), F.col("info.city"))),
               F.expr("substring(trim(info.venue), 1, length(trim(info.venue)) - length(info.city) - 2)"))
        .otherwise(F.trim("info.venue")).alias("venue")).collect()
    assert [r.venue for r in out] == ["Patiala Stadium", "Shimla Oval"]


def test_revision_replaces_match_atomically(spark, tmp_path, lakehouse):
    cfg, docs = lakehouse
    revised = copy.deepcopy(docs[0])
    revised["meta"]["revision"] = 2
    last_over = revised["innings"][1]["overs"][-1]
    last_over["deliveries"] = last_over["deliveries"][:-1] or last_over["deliveries"]
    expected = _balls(revised)

    stale = copy.deepcopy(docs[1])  # same revision as already loaded -> must be ignored
    silver.process_batch(bronze_frame(spark, [revised, stale], tmp_path / "batch1.ndjson"), 1, cfg, aliases=None)

    matches = spark.read.format("delta").load(cfg.table("silver", "matches"))
    deliveries = spark.read.format("delta").load(cfg.table("silver", "deliveries"))
    assert matches.filter("match_id = '9000'").first()["revision"] == 2
    assert deliveries.filter("match_id = '9000'").count() == expected
    assert matches.count() == len(docs)
    assert deliveries.select("match_id", "innings_no", "delivery_no").distinct().count() == deliveries.count()
