"""Gold feature layer: the match state after every ball, labelled with the final result.

gold.ball_state      one row per ball + one "start of innings" row (delivery_no = 0) per innings.
                     Feeds model training, batch win-probability scoring and WPA.
gold.venue_par_latest  rolling par score per venue, used by the live scorer.

Column definitions are mirrored by the streaming state machine (cricklake.streaming.state) and a unit
test asserts both produce identical features for the same match (train/serve parity).
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from cricklake.config import Settings, get_settings

log = logging.getLogger("cricklake.features")

FEATURES = [
    "innings_no", "legal_balls", "balls_remaining", "cum_runs", "cum_wickets", "wickets_in_hand", "run_rate",
    "runs_last_30", "wickets_last_30", "venue_par", "target", "runs_needed", "required_rate", "rate_gap",
    "par_gap", "is_female",
]
LABEL = "batting_team_won"
DEFAULT_PAR = 160.0
PAR_WINDOW = 20          # matches
RECENT_WINDOW = 30       # deliveries

IDENTITY = ["match_id", "match_date", "season_year", "event_name", "gender", "venue", "innings_no",
            "batting_team", "bowling_team", "winner", "result_type"]


def overs_to_balls(col):
    """Cricket notation: 16.4 overs = 16 overs and 4 balls = 100 balls."""
    return (F.floor(col) * 6 + F.round((col - F.floor(col)) * 10)).cast("int")


def venue_par(matches: DataFrame, deliveries: DataFrame) -> DataFrame:
    """As-of par score: mean first-innings total of the previous 20 completed innings at the venue.

    Strictly uses matches *before* the current one (rowsBetween(-20, -1)) so no future data leaks in.
    """
    first = (deliveries.filter("innings_no = 1").groupBy("match_id")
             .agg(F.sum("runs_total").alias("first_innings_runs"), F.max("legal_ball_no").alias("first_innings_balls")))
    wickets = deliveries.filter("innings_no = 1").groupBy("match_id").agg(
        F.sum(F.col("is_wicket").cast("int")).alias("first_innings_wickets"))
    m = (matches.select("match_id", "match_date", "venue", "gender", "event_name")
         .join(first, "match_id", "left").join(wickets, "match_id", "left"))
    complete = (F.col("first_innings_balls") >= 114) | (F.col("first_innings_wickets") >= 10)
    runs = F.when(complete, F.col("first_innings_runs"))
    order = ["match_date", "match_id"]
    w_venue = Window.partitionBy("venue", "gender").orderBy(*order).rowsBetween(-PAR_WINDOW, -1)
    w_event = Window.partitionBy("event_name").orderBy(*order).rowsBetween(-3 * PAR_WINDOW, -1)
    return (m.withColumn("_runs", runs)
            .withColumn("venue_avg", F.avg("_runs").over(w_venue))
            .withColumn("venue_n", F.count("_runs").over(w_venue))
            .withColumn("event_avg", F.avg("_runs").over(w_event))
            .withColumn("venue_par", F.when(F.col("venue_n") >= 5, F.col("venue_avg"))
                        .otherwise(F.coalesce(F.col("event_avg"), F.lit(DEFAULT_PAR))))
            .select("match_id", "venue", "gender", "event_name", "match_date", "_runs", "venue_par"))


def add_state_features(df: DataFrame) -> DataFrame:
    """Derive model features from the running totals (cum_runs, cum_wickets, legal_balls, last-30 window)."""
    inn2 = F.col("innings_no") == 2
    max_balls = F.when(inn2, F.coalesce(overs_to_balls(F.col("target_overs")), F.lit(120))) \
                 .otherwise(F.coalesce(F.col("scheduled_overs"), F.lit(20)) * 6)
    target = F.when(inn2, F.coalesce(F.col("target_runs"), F.col("first_innings_total") + 1))
    df = (
        df.withColumn("max_balls", max_balls.cast("int"))
        .withColumn("balls_remaining", F.greatest(F.col("max_balls") - F.col("legal_balls"), F.lit(0)))
        .withColumn("wickets_in_hand", 10 - F.col("cum_wickets"))
        .withColumn("run_rate", F.coalesce(F.try_divide(F.col("cum_runs") * 6.0, F.col("legal_balls")), F.lit(0.0)))
        .withColumn("venue_par", F.coalesce(F.col("venue_par"), F.lit(DEFAULT_PAR)).cast("double"))
        .withColumn("target", F.coalesce(target, F.lit(-1)).cast("int"))
        .withColumn("runs_needed", F.when(inn2, F.col("target") - F.col("cum_runs")).otherwise(F.lit(-1)))
    )
    rrr = F.when(F.col("runs_needed") <= 0, F.lit(0.0)).otherwise(
        F.least(F.coalesce(F.try_divide(F.col("runs_needed") * 6.0, F.col("balls_remaining")), F.lit(36.0)),
                F.lit(36.0)))
    return (
        df.withColumn("required_rate", F.when(inn2, rrr).otherwise(F.lit(-1.0)))
        .withColumn("rate_gap", F.when(inn2, F.col("required_rate") - F.col("run_rate")).otherwise(F.lit(0.0)))
        .withColumn("par_gap", F.when(inn2, F.col("target") - F.col("venue_par"))
                    .otherwise(F.col("cum_runs") - F.col("venue_par") * F.col("legal_balls") / F.col("max_balls")))
        .withColumn("is_female", F.coalesce((F.col("gender") == "female").cast("int"), F.lit(0)))
        .withColumn(LABEL, F.when(F.col("result_type").isin("no result", "draw") | F.col("winner").isNull(),
                                  F.lit(None).cast("int"))
                    .otherwise((F.col("winner") == F.col("batting_team")).cast("int")))
    )


def build_ball_state(spark: SparkSession, cfg: Settings) -> DataFrame:
    types = cfg["model"]["match_types"]
    # standard 20-over, 6-ball-over cricket only (real data also holds formats like The Hundred)
    matches = (spark.read.format("delta").load(cfg.table("silver", "matches"))
               .filter(F.col("match_type").isin(types) & (F.col("balls_per_over") == 6)
                       & (F.coalesce(F.col("scheduled_overs"), F.lit(20)) == 20)))
    deliveries = (spark.read.format("delta").load(cfg.table("silver", "deliveries"))
                  .filter(F.col("match_type").isin(types) & ~F.col("is_super_over") & F.col("innings_no").isin(1, 2)))
    par = venue_par(matches, deliveries)
    first_total = deliveries.filter("innings_no = 1").groupBy("match_id").agg(
        F.sum("runs_total").alias("first_innings_total"))

    base = (deliveries.drop("winner", "result_type")
            .join(F.broadcast(matches.select("match_id", "winner", "result_type", "scheduled_overs")), "match_id")
            .join(F.broadcast(first_total), "match_id", "left")
            .join(F.broadcast(par.select("match_id", "venue_par")), "match_id", "left"))

    seq = Window.partitionBy("match_id", "innings_no").orderBy("delivery_no")
    running = seq.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    recent = seq.rowsBetween(-(RECENT_WINDOW - 1), Window.currentRow)
    wicket = F.col("is_wicket").cast("int")
    context = ["target_runs", "target_overs", "scheduled_overs", "first_innings_total", "venue_par"]
    ball_cols = ["delivery_no", "over_no", "phase", "batter", "batter_id", "bowler", "bowler_id", "runs_batter",
                 "runs_total", "is_legal", "is_wicket", "is_four", "is_six", "bowler_wicket"]

    balls = base.select(
        *IDENTITY, *context, *ball_cols,
        F.sum("runs_total").over(running).cast("int").alias("cum_runs"),
        F.least(F.sum(wicket).over(running), F.lit(10)).cast("int").alias("cum_wickets"),
        F.col("legal_ball_no").cast("int").alias("legal_balls"),
        F.sum("runs_total").over(recent).cast("int").alias("runs_last_30"),
        F.sum(wicket).over(recent).cast("int").alias("wickets_last_30"),
        (F.col("delivery_no") == F.max("delivery_no").over(Window.partitionBy("match_id", "innings_no")))
        .alias("is_innings_end"),
    )
    starts = base.filter("delivery_no = 1").select(
        *IDENTITY, *context,
        F.lit(0).alias("delivery_no"), F.lit(0).alias("over_no"), F.lit("start").alias("phase"),
        F.lit(False).alias("is_innings_end"),
        *[F.lit(None).cast("string").alias(c) for c in ("batter", "batter_id", "bowler", "bowler_id")],
        F.lit(0).alias("runs_batter"), F.lit(0).alias("runs_total"),
        *[F.lit(False).alias(c) for c in ("is_legal", "is_wicket", "is_four", "is_six", "bowler_wicket")],
        *[F.lit(0).alias(c) for c in ("cum_runs", "cum_wickets", "legal_balls", "runs_last_30", "wickets_last_30")],
    )
    model_cfg = cfg["model"]
    split = (F.when(F.col("match_date") <= F.lit(model_cfg["train_until"]).cast("date"), "train")
             .when(F.col("match_date") <= F.lit(model_cfg["valid_until"]).cast("date"), "valid")
             .otherwise("test"))
    return add_state_features(balls.unionByName(starts)).withColumn("split", split)


def run(spark: SparkSession) -> dict:
    cfg = get_settings()
    state = build_ball_state(spark, cfg)
    (state.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
     .partitionBy("season_year").save(cfg.table("gold", "ball_state")))

    # latest par per venue for the live scorer (includes the most recent match, unlike the as-of par)
    matches = spark.read.format("delta").load(cfg.table("silver", "matches"))
    deliveries = spark.read.format("delta").load(cfg.table("silver", "deliveries")).filter(
        "NOT is_super_over AND innings_no IN (1, 2)")
    par = venue_par(matches, deliveries).filter(F.col("_runs").isNotNull())
    latest = Window.partitionBy("venue", "gender").orderBy(F.col("match_date").desc(), F.col("match_id").desc())
    (par.withColumn("_rn", F.row_number().over(latest))
     .filter(F.col("_rn") <= PAR_WINDOW)
     .groupBy("venue", "gender").agg(F.avg("_runs").alias("venue_par"), F.count("*").alias("innings_sample"))
     .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
     .save(cfg.table("gold", "venue_par_latest")))

    out = spark.read.format("delta").load(cfg.table("gold", "ball_state"))
    stats = {row["split"]: row["count"] for row in out.groupBy("split").count().collect()}
    log.info("gold.ball_state rows by split: %s", {k: f"{v:,}" for k, v in sorted(stats.items())})
    return {"ball_state_rows": sum(stats.values()), **{f"rows_{k}": v for k, v in stats.items()}}


if __name__ == "__main__":
    from cricklake.spark import get_spark

    run(get_spark("features"))
