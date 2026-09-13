"""Silver: validated, conformed, de-duplicated match and ball-by-ball tables.

bronze.matches --(stream, availableNow, foreachBatch)--> silver.matches     one row per match (latest revision)
                                                     --> silver.deliveries  one row per ball, partitioned by season
                                                     --> silver.quarantine  rejected records with a reason
Afterwards silver.players (player registry) and silver.team_aliases (reference data) are rebuilt.

Writes are Delta MERGEs keyed on natural keys, so a re-published revision atomically replaces the old
version of a match (including deleting balls that no longer exist) and replaying a batch is idempotent.
"""

from __future__ import annotations

import logging

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from cricklake.config import Settings, get_settings

log = logging.getLogger("cricklake.silver")

NOT_BOWLER_WICKETS = ("run out", "retired hurt", "retired out", "obstructing the field")


def _slug(col):
    return F.regexp_replace(F.regexp_replace(F.lower(col), r"[^a-z0-9]+", "_"), r"^_+|_+$", "")


def load_aliases(spark: SparkSession, cfg: Settings) -> DataFrame | None:
    path = cfg.landing / "reference" / "team_aliases.csv"
    if not path.exists():
        return None
    aliases = spark.read.option("header", True).csv(str(path))
    aliases.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        cfg.table("silver", "team_aliases"))
    return spark.read.format("delta").load(cfg.table("silver", "team_aliases"))


def conform_matches(df: DataFrame, aliases: DataFrame | None) -> DataFrame:
    """Flatten match-level info and normalise messy fields. Keeps raw innings for the ball explode."""
    venue = F.trim(F.col("info.venue"))
    city = F.col("info.city")
    outcome = F.col("info.outcome")
    m = df.select(
        "match_id",
        F.coalesce(F.col("meta.revision"), F.lit(1)).alias("revision"),
        F.col("meta.data_version").alias("data_version"),
        F.try_to_timestamp(F.try_element_at(F.col("info.dates"), F.lit(1)), F.lit("yyyy-MM-dd"))
        .cast("date").alias("match_date"),
        F.col("info.season").alias("season"),
        F.expr("try_cast(substring(info.season, 1, 4) AS INT)").alias("season_year"),
        # some real internationals carry no series name; a NULL key would break joins and uniqueness downstream
        F.coalesce(F.col("info.event.name"),
                   F.concat(F.initcap("info.team_type"), F.lit(" "), F.col("info.match_type"), F.lit(" (no event name)")),
                   F.lit("Unnamed event")).alias("event_name"),
        F.col("info.event.match_number").alias("match_number"),
        F.col("info.event.stage").alias("stage"),
        F.col("info.gender").alias("gender"),
        F.col("info.match_type").alias("match_type"),
        F.col("info.team_type").alias("team_type"),
        F.coalesce(F.col("info.balls_per_over"), F.lit(6)).alias("balls_per_over"),
        F.col("info.overs").alias("scheduled_overs"),
        # "X Stadium, City" and "X Stadium" are the same ground
        F.when(city.isNotNull() & venue.endswith(F.concat(F.lit(", "), city)),
               F.expr("substring(trim(info.venue), 1, length(trim(info.venue)) - length(info.city) - 2)"))
        .otherwise(venue).alias("venue"),
        city.alias("city"),
        F.try_element_at(F.col("info.teams"), F.lit(1)).alias("team_1"),
        F.try_element_at(F.col("info.teams"), F.lit(2)).alias("team_2"),
        F.col("info.toss.winner").alias("toss_winner"),
        F.col("info.toss.decision").alias("toss_decision"),
        F.coalesce(outcome.getField("winner"), outcome.getField("eliminator")).alias("winner"),
        F.coalesce(outcome.getField("result"),
                   F.when(outcome.getField("winner").isNotNull(), F.lit("win"))).alias("result_type"),
        outcome.getField("by").getField("runs").alias("win_by_runs"),
        outcome.getField("by").getField("wickets").alias("win_by_wickets"),
        outcome.getField("method").alias("method"),
        outcome.getField("eliminator").isNotNull().alias("decided_by_super_over"),
        F.try_element_at(F.col("info.player_of_match"), F.lit(1)).alias("player_of_match"),
        F.coalesce(F.size("innings"), F.lit(0)).alias("n_innings"),
        "innings",
        F.col("info.registry.people").alias("registry"),
        "_source_file",
        "_batch_id",
    )
    if aliases is not None:
        a = F.broadcast(aliases.select("event_name", "team_name", "franchise_id"))
        for side in ("team_1", "team_2"):
            m = m.join(a.withColumnRenamed("team_name", side).withColumnRenamed("franchise_id", f"{side}_franchise"),
                       ["event_name", side], "left")
    else:
        m = m.withColumn("team_1_franchise", F.lit(None).cast("string")) \
             .withColumn("team_2_franchise", F.lit(None).cast("string"))
    for side in ("team_1", "team_2"):
        m = m.withColumn(f"{side}_franchise",
                         F.coalesce(F.col(f"{side}_franchise"), _slug(F.concat_ws("-", "event_name", side))))
    return m


def explode_deliveries(matches: DataFrame) -> DataFrame:
    """innings[] -> overs[] -> deliveries[]  ==>  one row per ball with a stable ordering key."""
    base = ["match_id", "match_date", "season_year", "event_name", "gender", "match_type", "venue",
            "team_1", "team_2", "balls_per_over"]
    inn = matches.select(*base, "registry", F.posexplode("innings").alias("innings_idx", "inn"))
    ov = inn.select(
        *base, "registry",
        (F.col("innings_idx") + 1).alias("innings_no"),
        F.col("inn.team").alias("batting_team"),
        F.coalesce(F.col("inn.super_over"), F.lit(False)).alias("is_super_over"),
        F.col("inn.target.runs").alias("target_runs"),
        F.col("inn.target.overs").alias("target_overs"),
        F.explode("inn.overs").alias("ov"),
    )
    dl = ov.select("*", F.col("ov.over").alias("over_no"), F.posexplode("ov.deliveries").alias("ball_idx", "d"))

    wides = F.coalesce(F.col("d.extras.wides"), F.lit(0))
    noballs = F.coalesce(F.col("d.extras.noballs"), F.lit(0))
    n_wickets = F.when(F.col("d.wickets").isNull(), F.lit(0)).otherwise(F.size("d.wickets"))
    first_wicket = F.try_element_at(F.col("d.wickets"), F.lit(1))
    balls = dl.select(
        *base, "innings_no", "is_super_over", "batting_team",
        F.when(F.col("batting_team") == F.col("team_1"), F.col("team_2")).otherwise(F.col("team_1"))
        .alias("bowling_team"),
        "target_runs", "target_overs", "over_no", "ball_idx",
        F.col("d.batter").alias("batter"),
        F.col("d.bowler").alias("bowler"),
        F.col("d.non_striker").alias("non_striker"),
        F.try_element_at("registry", F.col("d.batter")).alias("batter_id"),
        F.try_element_at("registry", F.col("d.bowler")).alias("bowler_id"),
        F.col("d.runs.batter").alias("runs_batter"),
        F.col("d.runs.extras").alias("runs_extras"),
        F.col("d.runs.total").alias("runs_total"),
        wides.alias("wides"),
        noballs.alias("noballs"),
        F.coalesce(F.col("d.extras.byes"), F.lit(0)).alias("byes"),
        F.coalesce(F.col("d.extras.legbyes"), F.lit(0)).alias("legbyes"),
        ((wides == 0) & (noballs == 0)).alias("is_legal"),
        (n_wickets > 0).alias("is_wicket"),
        first_wicket.getField("kind").alias("wicket_kind"),
        first_wicket.getField("player_out").alias("player_out"),
        F.try_element_at("registry", first_wicket.getField("player_out")).alias("player_out_id"),
        ((F.col("d.runs.batter") == 4) & ~F.coalesce(F.col("d.runs.non_boundary"), F.lit(False))).alias("is_four"),
        (F.col("d.runs.batter") == 6).alias("is_six"),
    )
    seq = Window.partitionBy("match_id", "innings_no").orderBy("over_no", "ball_idx")
    return (
        balls.withColumn("delivery_no", F.row_number().over(seq))
        .withColumn("legal_ball_no", F.sum(F.col("is_legal").cast("int")).over(
            seq.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
        .withColumn("phase", F.when(F.col("over_no") < 6, "powerplay")
                    .when(F.col("over_no") < 15, "middle").otherwise("death"))
        .withColumn("bowler_wicket", F.col("is_wicket")
                    & ~F.coalesce(F.col("wicket_kind"), F.lit("")).isin(*NOT_BOWLER_WICKETS))
    )


def _quarantine(df: DataFrame, cfg: Settings, batch_id: int, stage: str) -> int:
    n = df.count()
    if n:
        (df.withColumn("quarantined_at", F.current_timestamp())
         .write.format("delta").mode("append")
         .option("txnAppId", f"silver_quarantine_{stage}").option("txnVersion", batch_id)  # idempotent retry
         .option("mergeSchema", "true")
         .save(cfg.table("silver", "quarantine")))
    return n


def _upsert(spark: SparkSession, df: DataFrame, path: str, keys: list[str], partition_by: list[str] | None = None,
            delete_unmatched_where: str | None = None) -> None:
    if not DeltaTable.isDeltaTable(spark, path):
        writer = df.write.format("delta").mode("overwrite")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)
        return
    condition = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    merge = (DeltaTable.forPath(spark, path).alias("t").merge(df.alias("s"), condition)
             .whenMatchedUpdateAll().whenNotMatchedInsertAll())
    if delete_unmatched_where:
        merge = merge.whenNotMatchedBySourceDelete(condition=delete_unmatched_where)
    merge.execute()


def process_batch(batch: DataFrame, batch_id: int, cfg: Settings, aliases: DataFrame | None) -> None:
    spark = batch.sparkSession
    batch = batch.persist()

    malformed = batch.filter(F.col("_corrupt_record").isNotNull() | F.col("match_id").isNull())
    n_malformed = _quarantine(malformed.select(
        "match_id", F.lit("MALFORMED_DOCUMENT").alias("reason"),
        F.substring("_corrupt_record", 1, 500).alias("detail"), "_source_file", "_batch_id"), cfg, batch_id, "doc")

    good = batch.filter(F.col("_corrupt_record").isNull() & F.col("match_id").isNotNull())
    newest_first = Window.partitionBy("match_id").orderBy(F.col("meta.revision").desc_nulls_last(),
                                                          F.col("_ingested_at").desc())
    latest = good.withColumn("_rn", F.row_number().over(newest_first)).filter("_rn = 1").drop("_rn")
    matches = conform_matches(latest, aliases)

    broken = F.col("match_date").isNull() | F.col("team_1").isNull() | F.col("team_2").isNull()
    n_broken = _quarantine(matches.filter(broken).select(
        "match_id", F.lit("MISSING_DATE_OR_TEAMS").alias("reason"), F.lit(None).cast("string").alias("detail"),
        "_source_file", "_batch_id"), cfg, batch_id, "match")
    matches = matches.filter(~broken)

    matches_path = cfg.table("silver", "matches")
    if DeltaTable.isDeltaTable(spark, matches_path):
        current = spark.read.format("delta").load(matches_path).select("match_id", F.col("revision").alias("_cur"))
        matches = (matches.join(current, "match_id", "left")
                   .filter(F.col("_cur").isNull() | (F.col("revision") > F.col("_cur"))).drop("_cur"))
    # Pin this micro-batch as a snapshot. A merely cached plan still reads silver.matches, and Spark re-evaluates
    # it as soon as we MERGE into silver.matches: the revision filter would then return nothing and the ball
    # MERGE below would delete every ball of the revised matches. (Caught by tests/test_silver.py.)
    matches = matches.localCheckpoint()
    changed_ids = [r.match_id for r in matches.select("match_id").collect()]
    if not changed_ids:
        log.info("batch %s: no new or revised matches", batch_id)
        batch.unpersist()
        return

    deliveries = explode_deliveries(matches)
    integrity_bad = (F.coalesce(F.col("runs_total") != F.col("runs_batter") + F.col("runs_extras"), F.lit(True))
                     | F.col("batter").isNull() | F.col("bowler").isNull())
    deliveries = deliveries.withColumn("_bad", integrity_bad).persist()
    n_bad_balls = _quarantine(deliveries.filter("_bad").select(
        "match_id", F.lit("DELIVERY_INTEGRITY").alias("reason"),
        F.to_json(F.struct("innings_no", "over_no", "ball_idx", "runs_batter", "runs_extras", "runs_total"))
        .alias("detail"), F.lit(None).cast("string").alias("_source_file"),
        F.lit(None).cast("string").alias("_batch_id")), cfg, batch_id, "ball")
    deliveries = deliveries.filter(~F.col("_bad")).drop("_bad")

    id_list = ", ".join(f"'{m}'" for m in changed_ids)
    # Balls first, match row last: the match row's revision is the commit marker. If the job dies in between,
    # the batch is replayed, still sees the old revision, and both idempotent MERGEs simply run again.
    _upsert(spark, deliveries, cfg.table("silver", "deliveries"), ["match_id", "innings_no", "delivery_no"],
            partition_by=["season_year"], delete_unmatched_where=f"t.match_id IN ({id_list})")
    _upsert(spark, matches.drop("innings", "registry"), matches_path, ["match_id"])
    log.info("batch %s: upserted %s matches; quarantined %d docs, %d matches, %d balls",
             batch_id, f"{len(changed_ids):,}", n_malformed, n_broken, n_bad_balls)
    deliveries.unpersist()
    batch.unpersist()


def rebuild_players(spark: SparkSession, cfg: Settings) -> int:
    d = spark.read.format("delta").load(cfg.table("silver", "deliveries"))
    cols = ["match_id", "match_date", "event_name", "gender"]
    people = (d.select(F.col("batter_id").alias("player_id"), F.col("batter").alias("player_name"),
                       F.col("batting_team").alias("team"), *cols)
              .unionByName(d.select(F.col("bowler_id").alias("player_id"), F.col("bowler").alias("player_name"),
                                    F.col("bowling_team").alias("team"), *cols))
              .filter(F.col("player_id").isNotNull()))
    players = people.groupBy("player_id").agg(
        F.max_by("player_name", "match_date").alias("player_name"),
        F.max_by("team", "match_date").alias("latest_team"),
        F.max_by("event_name", "match_date").alias("latest_event"),
        F.first("gender").alias("gender"),
        F.min("match_date").alias("first_match"),
        F.max("match_date").alias("last_match"),
        F.countDistinct("match_id").alias("matches"),
    )
    players.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        cfg.table("silver", "players"))
    return spark.read.format("delta").load(cfg.table("silver", "players")).count()


def run(spark: SparkSession) -> dict:
    cfg = get_settings()
    aliases = load_aliases(spark, cfg)
    stream = spark.readStream.format("delta").option("skipChangeCommits", "true").load(cfg.table("bronze", "matches"))
    query = (
        stream.writeStream.foreachBatch(lambda df, bid: process_batch(df, bid, cfg, aliases))
        .option("checkpointLocation", cfg.checkpoint("silver_matches"))
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    n_players = rebuild_players(spark, cfg)
    stats = {
        "silver_matches": spark.read.format("delta").load(cfg.table("silver", "matches")).count(),
        "silver_deliveries": spark.read.format("delta").load(cfg.table("silver", "deliveries")).count(),
        "silver_players": n_players,
    }
    log.info("Silver totals: %s", {k: f"{v:,}" for k, v in stats.items()})
    return stats


if __name__ == "__main__":
    from cricklake.spark import get_spark

    run(get_spark("silver"))
