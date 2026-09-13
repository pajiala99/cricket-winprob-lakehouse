"""Train/serve parity: streaming state + shared feature function == batch window functions."""

import pandas as pd
from pyspark.sql import functions as F

from cricklake.pipelines.features import FEATURES, add_state_features, build_ball_state, overs_to_balls
from cricklake.streaming.state import MatchState


def test_overs_notation(spark):
    row = spark.createDataFrame([(16.4,), (20.0,)], ["o"]).select(overs_to_balls(F.col("o")).alias("b")).collect()
    assert [r.b for r in row] == [100, 120]


def test_chase_features_for_known_state(spark):
    df = spark.createDataFrame(
        [(2, 150, 5, 90, 20, 5, 170.0, 181, 20.0, 180, 20, "male", "win", "A", "A")],
        "innings_no int, cum_runs int, cum_wickets int, legal_balls int, runs_last_30 int, wickets_last_30 int, "
        "venue_par double, target_runs int, target_overs double, first_innings_total int, scheduled_overs int, "
        "gender string, result_type string, winner string, batting_team string")
    r = add_state_features(df).first()
    assert r.balls_remaining == 30 and r.runs_needed == 31 and r.wickets_in_hand == 5
    assert abs(r.required_rate - 6.2) < 1e-9 and abs(r.run_rate - 10.0) < 1e-9
    assert r.par_gap == 11.0 and r.batting_team_won == 1


def test_streaming_state_reproduces_batch_features(spark, lakehouse):
    cfg, _ = lakehouse
    batch = build_ball_state(spark, cfg).filter("delivery_no >= 1").toPandas()
    deliveries = (spark.read.format("delta").load(cfg.table("silver", "deliveries"))
                  .filter("NOT is_super_over AND innings_no IN (1, 2)").toPandas())

    rows = []
    for _, balls in deliveries.groupby("match_id"):
        state = MatchState()
        for ball in balls.sort_values(["innings_no", "delivery_no"]).to_dict("records"):
            totals = state.apply(ball)
            assert state.apply(ball) is None  # replayed event is ignored
            rows.append({"match_id": ball["match_id"], "innings_no": ball["innings_no"],
                         "delivery_no": ball["delivery_no"], **totals,
                         "first_innings_total": state.first_innings_total if ball["innings_no"] == 2 else None})
    stream = pd.DataFrame(rows)

    context = batch[["match_id", "innings_no", "delivery_no", "venue_par", "target_runs", "target_overs",
                     "scheduled_overs", "gender", "result_type", "winner", "batting_team"]]
    merged = stream.merge(context, on=["match_id", "innings_no", "delivery_no"], validate="one_to_one")
    assert len(merged) == len(batch)

    online = add_state_features(spark.createDataFrame(merged)).toPandas()
    joined = online.merge(batch, on=["match_id", "innings_no", "delivery_no"], suffixes=("_online", "_batch"))
    for feature in FEATURES:
        if feature == "innings_no":
            continue
        diff = (joined[f"{feature}_online"].astype(float) - joined[f"{feature}_batch"].astype(float)).abs().max()
        assert diff < 1e-9, f"{feature} differs between streaming and batch (max diff {diff})"
