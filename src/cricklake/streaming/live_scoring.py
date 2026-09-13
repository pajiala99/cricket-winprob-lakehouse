"""Real-time win probability with Spark Structured Streaming.

inbox/*/*.json
  -> readStream (explicit schema)
  -> groupBy(match_id).applyInPandasWithState   per-match running totals, duplicate-ball protection, state TTL
  -> foreachBatch:
       join the rolling venue par, derive features with the *batch* feature function,
       score with the batch-trained model, pin finished chases, compute ΔWP against the previous ball,
       append gold.live_win_probability and flag momentum swings in gold.live_swing_alerts

  python -m cricklake.streaming.live_scoring            # run continuously
  python -m cricklake.streaming.live_scoring --once     # drain the inbox and stop (tests / demos)
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout
from pyspark.sql.types import IntegerType, StructField, StructType

from cricklake.config import get_settings
from cricklake.ml.train import MODEL_DIR
from cricklake.pipelines.features import add_state_features
from cricklake.pipelines.impact import score
from cricklake.schemas import BALL_EVENT
from cricklake.streaming.state import MatchState

log = logging.getLogger("cricklake.live_scoring")

PASSTHROUGH = ["match_id", "event_name", "season_year", "gender", "venue", "innings_no", "delivery_no", "over_no",
               "batting_team", "bowling_team", "batter", "bowler", "runs_total", "is_wicket", "wicket_kind",
               "event_ts", "target_runs", "target_overs"]
TOTALS = ["cum_runs", "cum_wickets", "legal_balls", "runs_last_30", "wickets_last_30"]
_types = {f.name: f.dataType for f in BALL_EVENT.fields}
RUNNING = StructType(
    [StructField(c, _types[c]) for c in PASSTHROUGH]
    + [StructField("first_innings_total", IntegerType())]
    + [StructField(c, IntegerType()) for c in TOTALS]
)
STATE_TTL_MS = 30 * 60 * 1000


def update_match(key: tuple, batches: Iterator[pd.DataFrame], state: GroupState) -> Iterator[pd.DataFrame]:
    """Stateful step, executed per match_id on the executors."""
    if state.hasTimedOut:
        state.remove()  # match finished / feed went quiet: free the state
        return
    match = MatchState.from_json(state.get[0] if state.exists else None)
    balls = pd.concat(list(batches)).sort_values(["innings_no", "delivery_no"])
    rows = []
    for ball in balls.to_dict("records"):
        totals = match.apply(ball)
        if totals is None:
            continue
        first_total = match.first_innings_total if int(ball["innings_no"]) == 2 else None
        rows.append({**{c: ball[c] for c in PASSTHROUGH}, "first_innings_total": first_total, **totals})
    state.update((match.to_json(),))
    state.setTimeoutDuration(STATE_TTL_MS)
    if rows:
        yield pd.DataFrame(rows, columns=RUNNING.fieldNames())


class LiveScorer:
    """foreachBatch sink. Runs on the driver, so it can keep the previous WP of every live innings in memory."""

    def __init__(self, spark: SparkSession, swing_threshold: float):
        self.cfg = get_settings()
        self.model = PipelineModel.load(str(self.cfg.models / MODEL_DIR))
        self.par = (spark.read.format("delta").load(self.cfg.table("gold", "venue_par_latest"))
                    .select("venue", "gender", "venue_par"))
        self.threshold = swing_threshold
        self.last_wp: dict[tuple[str, int], float] = {}

    def __call__(self, batch: DataFrame, batch_id: int) -> None:
        if batch.isEmpty():
            return
        enriched = (batch.join(F.broadcast(self.par), ["venue", "gender"], "left")
                    .withColumn("scheduled_overs", F.lit(20))
                    .withColumn("winner", F.lit(None).cast("string"))
                    .withColumn("result_type", F.lit(None).cast("string")))
        scored = score(self.model, add_state_features(enriched))
        selected = scored.select(*PASSTHROUGH, *TOTALS, "balls_remaining", "target", "required_rate", "run_rate",
                                 "model_win_prob", "win_prob")
        pdf = selected.toPandas().sort_values(["match_id", "innings_no", "delivery_no"])
        # WP before an innings' first ball = the model's score of the empty scoreboard, exactly like the
        # delivery_no = 0 rows of the batch table, so live ΔWP matches batch ΔWP
        starts = enriched.filter("delivery_no = 1").withColumns({c: F.lit(0) for c in TOTALS})
        start_wp = {(r.match_id, int(r.innings_no)): float(r.win_prob) for r in
                    score(self.model, add_state_features(starts)).select("match_id", "innings_no", "win_prob").collect()}

        before = []
        for r in pdf.itertuples():
            key = (r.match_id, int(r.innings_no))
            prev = self.last_wp.get(key, start_wp.get(key))
            if prev is None:  # scorer restarted mid-innings: best available estimate
                prev = 0.5 if r.innings_no == 1 else 1.0 - self.last_wp.get((r.match_id, 1), 0.5)
            before.append(prev)
            self.last_wp[key] = float(r.win_prob)
        pdf["wp_before"] = before
        pdf["delta_wp"] = pdf["win_prob"] - pdf["wp_before"]
        pdf["is_swing"] = pdf["delta_wp"].abs() >= self.threshold
        pdf["scored_at"] = pd.Timestamp.now(tz="UTC")

        spark = batch.sparkSession
        # explicit schema: a micro-batch without wickets has an all-null wicket_kind column that can't be inferred
        schema = (StructType.fromJson(selected.schema.jsonValue()).add("wp_before", "double")
                  .add("delta_wp", "double").add("is_swing", "boolean").add("scored_at", "timestamp"))
        out = spark.createDataFrame(pdf, schema=schema)
        (out.write.format("delta").mode("append").option("mergeSchema", "true")
         .option("txnAppId", "live_win_probability").option("txnVersion", batch_id)
         .save(self.cfg.table("gold", "live_win_probability")))
        swings = pdf[pdf["is_swing"]]
        if len(swings):
            (spark.createDataFrame(swings, schema=schema).write.format("delta").mode("append")
             .option("mergeSchema", "true")
             .option("txnAppId", "live_swing_alerts").option("txnVersion", batch_id)
             .save(self.cfg.table("gold", "live_swing_alerts")))

        last = pdf.iloc[-1]
        chase = (f" | need {int(last.target - last.cum_runs)} off {int(last.balls_remaining)}"
                 if last.innings_no == 2 and last.target > 0 else "")
        log.info("batch %d: %d balls | %s %d/%d (%d.%d ov)%s | %s win prob %.1f%% (%+.1f) | swings: %d",
                 batch_id, len(pdf), last.batting_team, last.cum_runs, last.cum_wickets, last.legal_balls // 6,
                 last.legal_balls % 6, chase, last.batting_team, 100 * last.win_prob, 100 * last.delta_wp, len(swings))


def build_query(spark: SparkSession):
    cfg = get_settings()
    stream_cfg = cfg["streaming"]
    events = (spark.readStream.schema(BALL_EVENT).option("maxFilesPerTrigger", 2000)
              .json(str(cfg.stream_inbox / "*" / "*.json")))
    running = events.groupBy("match_id").applyInPandasWithState(
        update_match, RUNNING, "state STRING", "append", GroupStateTimeout.ProcessingTimeTimeout)
    return (running.writeStream.foreachBatch(LiveScorer(spark, stream_cfg["swing_alert_threshold"]))
            .option("checkpointLocation", cfg.checkpoint("live_scoring"))
            .trigger(processingTime=f"{stream_cfg['trigger_seconds']} seconds")
            .start())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Score live ball events with Structured Streaming")
    ap.add_argument("--once", action="store_true", help="process everything in the inbox, then stop")
    args = ap.parse_args(argv)
    from cricklake.spark import get_spark

    spark = get_spark("live")
    if args.once:
        # State timeouts make Spark schedule "no-data" micro-batches forever, so neither availableNow nor
        # processAllAvailable() ever sees the inbox as drained. A one-off drain doesn't need them; the
        # continuous scorer keeps them, because they are what evicts finished matches from state.
        spark.conf.set("spark.sql.streaming.noDataMicroBatches.enabled", "false")
    query = build_query(spark)
    log.info("Live scorer started (query id %s)", query.id)
    if not args.once:
        query.awaitTermination()
        return
    query.processAllAvailable()
    balls = sum(p["numInputRows"] for p in query.recentProgress)
    batches = sum(1 for p in query.recentProgress if p["numInputRows"])
    query.stop()
    log.info("Inbox drained: %d ball event(s) in %d micro-batch(es)", balls, batches)
    spark.stop()


if __name__ == "__main__":
    main()
