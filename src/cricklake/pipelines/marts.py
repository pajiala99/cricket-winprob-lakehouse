"""Classic analytics marts built straight from silver (no model involved).

gold.team_standings        league table per season: played/won/lost/NR, points, Net Run Rate, champion
gold.venue_stats           per ground: average first-innings score, chase success, toss impact
gold.season_scoring        average first-innings score per competition and season (era inflation)
gold.batting_phase_stats   strike rate, boundary % and dot % by phase (powerplay / middle / death)
gold.bowling_phase_stats   economy, dot % and wickets by phase
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from cricklake.config import get_settings
from cricklake.pipelines.features import overs_to_balls

log = logging.getLogger("cricklake.marts")


def team_standings(matches: DataFrame, deliveries: DataFrame) -> DataFrame:
    league = matches.filter("stage IS NULL")
    innings = (deliveries.filter("innings_no IN (1, 2) AND NOT is_super_over")
               .join(league.filter("result_type <> 'no result'").select("match_id", "scheduled_overs"), "match_id")
               .groupBy("match_id", "event_name", "season_year", "innings_no", "batting_team", "bowling_team",
                        "scheduled_overs")
               .agg(F.sum("runs_total").alias("runs"), F.max("legal_ball_no").alias("balls"),
                    F.sum(F.col("is_wicket").cast("int")).alias("wickets"), F.max("target_overs").alias("target_overs")))
    quota = (F.when(F.col("innings_no") == 2, F.coalesce(overs_to_balls(F.col("target_overs")), F.lit(120)))
             .otherwise(F.coalesce(F.col("scheduled_overs"), F.lit(20)) * 6))
    # NRR rule: a side bowled out is charged its full quota of overs
    innings = innings.withColumn("nrr_balls", F.when(F.col("wickets") >= 10, quota).otherwise(F.col("balls")))
    keys = ["event_name", "season_year"]
    runs_rates = (
        innings.select(*keys, F.col("batting_team").alias("team"), F.col("runs").alias("runs_for"),
                       F.col("nrr_balls").alias("balls_for"), F.lit(0).alias("runs_against"),
                       F.lit(0).alias("balls_against"))
        .unionByName(innings.select(*keys, F.col("bowling_team").alias("team"), F.lit(0).alias("runs_for"),
                                    F.lit(0).alias("balls_for"), F.col("runs").alias("runs_against"),
                                    F.col("nrr_balls").alias("balls_against")))
        .groupBy(*keys, "team").agg(*[F.sum(c).alias(c) for c in ("runs_for", "balls_for", "runs_against",
                                                                  "balls_against")])
        .withColumn("net_run_rate", F.try_divide(F.col("runs_for") * 6.0, F.col("balls_for"))
                    - F.try_divide(F.col("runs_against") * 6.0, F.col("balls_against")))
    )
    results = (league.select(*keys, F.explode(F.array("team_1", "team_2")).alias("team"), "winner", "result_type")
               .groupBy(*keys, "team").agg(
                   F.count("*").alias("played"),
                   F.sum((F.col("winner") == F.col("team")).cast("int")).alias("won"),
                   F.sum((F.col("result_type") == "no result").cast("int")).alias("no_result")))
    champions = matches.filter("stage = 'Final'").select(*keys, F.col("winner").alias("team"),
                                                          F.lit(True).alias("is_champion"))
    rank = Window.partitionBy(*keys).orderBy(F.col("points").desc(), F.col("net_run_rate").desc())
    return (results.join(runs_rates, [*keys, "team"], "left")
            .join(champions, [*keys, "team"], "left")
            .withColumn("lost", F.col("played") - F.col("won") - F.col("no_result"))
            .withColumn("points", 2 * F.col("won") + F.col("no_result"))
            .withColumn("is_champion", F.coalesce(F.col("is_champion"), F.lit(False)))
            .withColumn("position", F.row_number().over(rank))
            .select(*keys, "position", "team", "played", "won", "lost", "no_result", "points",
                    F.round("net_run_rate", 3).alias("net_run_rate"), "is_champion"))


def venue_stats(matches: DataFrame, deliveries: DataFrame) -> tuple[DataFrame, DataFrame]:
    base = deliveries.filter("NOT is_super_over")
    first = base.filter("innings_no = 1").groupBy("match_id").agg(F.sum("runs_total").alias("first_innings_runs"))
    chasing = base.filter("innings_no = 2").groupBy("match_id").agg(F.first("batting_team").alias("chasing_team"))
    decided = (matches.filter("result_type IN ('win', 'tie')")
               .join(first, "match_id").join(chasing, "match_id"))
    venues = (decided.groupBy("venue", "gender").agg(
        F.count("*").alias("matches"),
        F.round(F.avg("first_innings_runs"), 1).alias("avg_first_innings"),
        F.max("first_innings_runs").alias("highest_first_innings"),
        F.round(100 * F.avg((F.col("winner") == F.col("chasing_team")).cast("double")), 1).alias("chase_win_pct"),
        F.round(100 * F.avg((F.col("winner") == F.col("toss_winner")).cast("double")), 1).alias("toss_winner_win_pct"),
        F.round(100 * F.avg((F.col("toss_decision") == "field").cast("double")), 1).alias("chose_to_field_pct"),
    ).filter("matches >= 10"))
    seasons = decided.groupBy("event_name", "gender", "season_year").agg(
        F.count("*").alias("matches"),
        F.round(F.avg("first_innings_runs"), 1).alias("avg_first_innings"),
        F.round(100 * F.avg((F.col("winner") == F.col("chasing_team")).cast("double")), 1).alias("chase_win_pct"))
    return venues, seasons


def phase_stats(deliveries: DataFrame) -> tuple[DataFrame, DataFrame]:
    d = deliveries.filter("NOT is_super_over")
    batting = (d.filter(F.col("batter_id").isNotNull())
               .groupBy(F.col("batter_id").alias("player_id"), "gender", "phase").agg(
                   F.max_by("batter", "match_date").alias("player_name"),
                   F.sum("runs_batter").alias("runs"),
                   F.sum(F.col("is_legal").cast("int")).alias("balls"),
                   F.sum((F.col("player_out_id") == F.col("batter_id")).cast("int")).alias("dismissals"),
                   F.sum((F.col("is_four") | F.col("is_six")).cast("int")).alias("boundaries"),
                   F.sum(((F.col("runs_batter") == 0) & F.col("is_legal")).cast("int")).alias("dots"))
               .withColumn("strike_rate", F.round(F.try_divide(F.col("runs") * 100.0, F.col("balls")), 1))
               .withColumn("boundary_pct", F.round(F.try_divide(F.col("boundaries") * 100.0, F.col("balls")), 1))
               .withColumn("dot_pct", F.round(F.try_divide(F.col("dots") * 100.0, F.col("balls")), 1)))
    conceded = F.col("runs_batter") + F.col("wides") + F.col("noballs")
    bowling = (d.filter(F.col("bowler_id").isNotNull())
               .groupBy(F.col("bowler_id").alias("player_id"), "gender", "phase").agg(
                   F.max_by("bowler", "match_date").alias("player_name"),
                   F.sum(conceded).alias("runs_conceded"),
                   F.sum(F.col("is_legal").cast("int")).alias("balls"),
                   F.sum(F.col("bowler_wicket").cast("int")).alias("wickets"),
                   F.sum(((F.col("runs_total") == 0) & F.col("is_legal")).cast("int")).alias("dots"))
               .withColumn("economy", F.round(F.try_divide(F.col("runs_conceded") * 6.0, F.col("balls")), 2))
               .withColumn("dot_pct", F.round(F.try_divide(F.col("dots") * 100.0, F.col("balls")), 1)))
    return batting, bowling


def run(spark: SparkSession) -> dict:
    cfg = get_settings()
    types = cfg["model"]["match_types"]
    matches = spark.read.format("delta").load(cfg.table("silver", "matches")).filter(F.col("match_type").isin(types))
    deliveries = (spark.read.format("delta").load(cfg.table("silver", "deliveries"))
                  .filter(F.col("match_type").isin(types)))
    venues, seasons = venue_stats(matches, deliveries)
    batting, bowling = phase_stats(deliveries)
    outputs = {
        "team_standings": team_standings(matches, deliveries),
        "venue_stats": venues,
        "season_scoring": seasons,
        "batting_phase_stats": batting,
        "bowling_phase_stats": bowling,
    }
    counts = {}
    for name, df in outputs.items():
        path = cfg.table("gold", name)
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
        counts[name] = spark.read.format("delta").load(path).count()
    log.info("Marts written: %s", counts)
    return counts


if __name__ == "__main__":
    from cricklake.spark import get_spark

    run(get_spark("marts"))
