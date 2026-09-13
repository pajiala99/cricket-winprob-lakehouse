"""Win probability for every ball, Win Probability Added (WPA), player impact and match excitement.

gold.win_probability   WP before/after each ball, ΔWP, leverage index, batter (+ΔWP) and bowler (-ΔWP) credit
gold.player_impact     per player-season: batting/bowling WPA, "clutch" WPA on high-leverage balls, classic stats
gold.match_excitement  per match: excitement index (Σ|ΔWP|), biggest swing ball, great-escape flag

WPA answers "how much did this player change his team's chance of winning?", which runs and wickets alone
cannot: a single off the last ball of a tight chase is worth more than a boundary in a dead rubber.
"""

from __future__ import annotations

import logging

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from cricklake.config import get_settings
from cricklake.ml.train import MODEL_DIR

log = logging.getLogger("cricklake.impact")
HIGH_LEVERAGE = 2.0
MODEL_OUTPUT_COLS = ("features", "scaled_features", "rawPrediction", "probability", "prediction")


def apply_terminal_states(df: DataFrame) -> DataFrame:
    """The model estimates uncertain states; a finished chase is certain, so pin it to 1, 0.5 (tie) or 0.

    A chase normally ends the moment the target is reached, except in rain-affected (D/L) games, where the
    recorded target is the par score when play was abandoned and can be passed mid-innings. The batch table
    therefore pins only the innings' final ball; live scoring (no is_innings_end column) pins at the target.
    """
    inn2 = F.col("innings_no") == 2
    ended = F.col("is_innings_end") if "is_innings_end" in df.columns else F.lit(True)
    finished = (F.col("wickets_in_hand") <= 0) | (F.col("balls_remaining") <= 0)
    return df.withColumn(
        "win_prob",
        F.when(inn2 & ended & (F.col("cum_runs") >= F.col("target")), F.lit(1.0))
        .when(inn2 & finished & (F.col("cum_runs") == F.col("target") - 1), F.lit(0.5))
        .when(inn2 & finished, F.lit(0.0))
        .otherwise(F.col("model_win_prob")),
    )


def score(model: PipelineModel, state: DataFrame) -> DataFrame:
    pred = model.transform(state).withColumn("model_win_prob", vector_to_array("probability").getItem(1))
    return apply_terminal_states(pred.drop(*MODEL_OUTPUT_COLS))


def win_probability(spark: SparkSession, model: PipelineModel, state: DataFrame) -> DataFrame:
    scored = score(model, state)
    seq = Window.partitionBy("match_id", "innings_no").orderBy("delivery_no")
    bucket = F.concat_ws("|", *[c.cast("string") for c in (
        F.col("innings_no"), F.col("over_no"), F.least(F.floor(F.col("cum_wickets") / 3), F.lit(3)),
        F.when(F.col("innings_no") == 2, F.least(F.floor(F.col("required_rate") / 2), F.lit(9))).otherwise(F.lit(-1)),
    )])
    balls = (scored.withColumn("wp_before", F.lag("win_prob").over(seq))
             .filter("delivery_no >= 1")
             .withColumn("delta_wp", F.col("win_prob") - F.col("wp_before"))
             .withColumn("batter_wpa", F.col("delta_wp"))
             .withColumn("bowler_wpa", -F.col("delta_wp"))
             .withColumn("lev_bucket", bucket)
             .persist())
    mean_abs = balls.select(F.avg(F.abs("delta_wp"))).first()[0]
    # leverage index: how much WP typically moves in this game situation, relative to an average ball
    leverage = balls.groupBy("lev_bucket").agg((F.avg(F.abs("delta_wp")) / F.lit(mean_abs)).alias("leverage_index"))
    return (balls.join(F.broadcast(leverage), "lev_bucket")
            .withColumn("is_high_leverage", F.col("leverage_index") >= HIGH_LEVERAGE)
            .select("match_id", "match_date", "season_year", "event_name", "gender", "venue", "split",
                    "innings_no", "delivery_no", "over_no", "phase", "batting_team", "bowling_team", "winner",
                    "result_type", "batter", "batter_id", "bowler", "bowler_id", "runs_batter", "runs_total",
                    "is_legal", "is_wicket", "is_four", "is_six", "bowler_wicket", "cum_runs", "cum_wickets",
                    "legal_balls", "balls_remaining", "target", "run_rate", "required_rate", "venue_par",
                    "model_win_prob", "wp_before", "win_prob", "delta_wp", "batter_wpa", "bowler_wpa",
                    "leverage_index", "is_high_leverage"))


def player_impact(wp: DataFrame) -> DataFrame:
    keys = ["event_name", "season_year", "gender"]
    hl = F.col("is_high_leverage")
    bat = (wp.filter(F.col("batter_id").isNotNull())
           .groupBy(F.col("batter_id").alias("player_id"), *keys).agg(
               F.max_by("batter", "match_date").alias("bat_name"),
               F.max_by("batting_team", "match_date").alias("bat_team"),
               F.countDistinct("match_id").alias("bat_matches"),
               F.sum("batter_wpa").alias("batting_wpa"),
               F.sum(F.when(hl, F.col("batter_wpa")).otherwise(F.lit(0.0))).alias("clutch_batting_wpa"),
               F.sum(hl.cast("int")).alias("hl_balls_faced"),
               F.sum("runs_batter").alias("runs"),
               F.sum(F.col("is_legal").cast("int")).alias("balls_faced"),
               F.sum((F.col("is_four") | F.col("is_six")).cast("int")).alias("boundaries")))
    bowl = (wp.filter(F.col("bowler_id").isNotNull())
            .groupBy(F.col("bowler_id").alias("player_id"), *keys).agg(
                F.max_by("bowler", "match_date").alias("bowl_name"),
                F.max_by("bowling_team", "match_date").alias("bowl_team"),
                F.countDistinct("match_id").alias("bowl_matches"),
                F.sum("bowler_wpa").alias("bowling_wpa"),
                F.sum(F.when(hl, F.col("bowler_wpa")).otherwise(F.lit(0.0))).alias("clutch_bowling_wpa"),
                F.sum(hl.cast("int")).alias("hl_balls_bowled"),
                F.sum("runs_total").alias("runs_conceded"),
                F.sum(F.col("is_legal").cast("int")).alias("balls_bowled"),
                F.sum(F.col("bowler_wicket").cast("int")).alias("wickets")))
    ints = ["hl_balls_faced", "runs", "balls_faced", "boundaries", "hl_balls_bowled", "runs_conceded",
            "balls_bowled", "wickets"]
    dbls = ["batting_wpa", "clutch_batting_wpa", "bowling_wpa", "clutch_bowling_wpa"]
    joined = bat.join(bowl, ["player_id", *keys], "full_outer")
    return (joined.select(
                "player_id", *keys,
                F.coalesce("bat_name", "bowl_name").alias("player_name"),
                F.coalesce("bat_team", "bowl_team").alias("team"),
                F.greatest(F.coalesce("bat_matches", F.lit(0)), F.coalesce("bowl_matches", F.lit(0))).alias("matches"),
                *[F.coalesce(c, F.lit(0)).cast("long").alias(c) for c in ints],
                *[F.coalesce(c, F.lit(0.0)).alias(c) for c in dbls])
            .withColumn("total_wpa", F.col("batting_wpa") + F.col("bowling_wpa"))
            .withColumn("clutch_wpa", F.col("clutch_batting_wpa") + F.col("clutch_bowling_wpa"))
            .withColumn("strike_rate", F.try_divide(F.col("runs") * 100.0, F.col("balls_faced")))
            .withColumn("economy", F.try_divide(F.col("runs_conceded") * 6.0, F.col("balls_bowled"))))


def match_excitement(wp: DataFrame) -> DataFrame:
    winner_wp = (F.when(F.col("batting_team") == F.col("winner"), F.col("win_prob"))
                 .when(F.col("bowling_team") == F.col("winner"), 1 - F.col("win_prob")))
    swing = F.struct("innings_no", "over_no", "delivery_no", "batter", "bowler", "runs_total", "is_wicket", "delta_wp")
    agg = wp.groupBy("match_id", "match_date", "season_year", "event_name", "gender", "venue", "winner",
                     "result_type").agg(
        F.sum(F.abs("delta_wp")).alias("excitement_index"),
        F.max(F.abs("delta_wp")).alias("biggest_swing"),
        F.max_by(swing, F.abs("delta_wp")).alias("swing"),
        F.min(winner_wp).alias("winner_lowest_wp"),
        F.first(F.when(F.col("innings_no") == 1, F.col("batting_team")), ignorenulls=True).alias("batting_first"),
        F.first(F.when(F.col("innings_no") == 2, F.col("batting_team")), ignorenulls=True).alias("chasing"),
    )
    return agg.select(
        "*",
        *[F.col(f"swing.{c}").alias(f"swing_{c}") for c in
          ("innings_no", "over_no", "delivery_no", "batter", "bowler", "runs_total", "is_wicket", "delta_wp")],
        (F.col("winner_lowest_wp") < 0.10).alias("is_great_escape"),
    ).drop("swing")


def run(spark: SparkSession) -> dict:
    cfg = get_settings()
    model = PipelineModel.load(str(cfg.models / MODEL_DIR))
    state = spark.read.format("delta").load(cfg.table("gold", "ball_state"))
    wp_path = cfg.table("gold", "win_probability")
    (win_probability(spark, model, state).write.format("delta").mode("overwrite").option("overwriteSchema", "true")
     .partitionBy("season_year").save(wp_path))
    wp = spark.read.format("delta").load(wp_path)

    for name, df in (("player_impact", player_impact(wp)), ("match_excitement", match_excitement(wp))):
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(cfg.table("gold", name))
    stats = {
        "win_probability_rows": wp.count(),
        "player_seasons": spark.read.format("delta").load(cfg.table("gold", "player_impact")).count(),
        "great_escapes": spark.read.format("delta").load(cfg.table("gold", "match_excitement"))
        .filter("is_great_escape").count(),
    }
    log.info("Impact layer: %s", {k: f"{v:,}" for k, v in stats.items()})
    return stats


if __name__ == "__main__":
    from cricklake.spark import get_spark

    run(get_spark("impact"))
