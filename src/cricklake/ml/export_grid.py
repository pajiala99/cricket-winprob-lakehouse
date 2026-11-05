"""Precompute the win-probability model's predictions over a grid of chase states, for the portfolio widget.

    CRICKLAKE_CONFIG=conf/real.yaml python -m cricklake.ml.export_grid --copy-to ../hunter-portfolio/assets/cricket

A chase state is overs left (1-20) x wickets in hand (1-10) x runs needed. Each state is scored by the trained
champion through the same add_state_features and score functions the pipeline uses, so the widget only looks
numbers up: nothing is interpolated or smoothed.

The model also reads inputs a slider can't set: runs scored so far, the last five overs, the ground's par score.
Those are filled in with what real chases looked like: the median score and recent form of men's training-set
chases at the same over with the same wickets down, at a ground with the median test-season par. Then
target = runs so far + runs needed. Filling them with an invented "par pace" instead created states no real chase
reaches (64/8 after eight overs), where the trees extrapolate wildly, so every state also records how many real
training chases passed within five runs of it, and the widget says when that number is small.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from cricklake.config import get_settings
from cricklake.ml.train import MODEL_DIR
from cricklake.pipelines.features import RECENT_WINDOW, add_state_features
from cricklake.pipelines.impact import score

log = logging.getLogger("cricklake.export_grid")
GRID_NAME = "winprob_grid.json"
MAX_NEED = 250
NEAR_RUNS = 5                   # support counts chases within +-5 runs needed of a state
SUPPORT_LEVELS = (5, 25, 100)   # level 0: < 5 real chases nearby, 1: 5-24, 2: 25-99, 3: 100+
MIN_PAST = 10                   # fewer real chases than this at an (over, wickets) point: use par pace instead


def max_need(overs_left: int) -> int:
    """Largest runs-needed value worth storing: six sixes an over plus a little for extras, capped."""
    return min(MAX_NEED, 36 * overs_left + 6)


def chase_over_ends(spark: SparkSession, cfg) -> DataFrame:
    """Men's 20-over training chases, one row per match at the end of every over (plus the start of the chase)."""
    state = F.struct("cum_runs", "runs_last_30", "wickets_last_30", "runs_needed", "wickets_in_hand",
                     "balls_remaining")
    return (spark.read.format("delta").load(cfg.table("gold", "ball_state"))
            .filter("innings_no = 2 AND is_female = 0 AND max_balls = 120 AND split = 'train' AND legal_balls % 6 = 0")
            .groupBy("match_id", "legal_balls").agg(F.max_by(state, "delivery_no").alias("s"))
            .select("match_id", "s.*")
            .filter("balls_remaining >= 6 AND wickets_in_hand >= 1 AND runs_needed >= 1")
            .withColumn("overs_left", (F.col("balls_remaining") / 6).cast("int")))


def median_par(spark: SparkSession, cfg) -> tuple[float, dict]:
    """Median as-of venue par of men's chases in each split; the test seasons' median is the widget's ground."""
    starts = (spark.read.format("delta").load(cfg.table("gold", "ball_state"))
              .filter("innings_no = 2 AND delivery_no = 0 AND is_female = 0"))
    by_split = {s: starts.filter(F.col("split") == s).approxQuantile("venue_par", [0.5], 0.0)[0]
                for s in ("train", "valid", "test")}
    return round(by_split["test"]), by_split


def chase_states(par: float, past: dict) -> tuple[list[tuple], dict]:
    """Every (overs left, wickets in hand, runs needed) state, with the model's other inputs filled in."""
    rate, rows, used = par / 120, [], {}
    for overs_left in range(1, 21):
        legal = 120 - 6 * overs_left
        recent = min(RECENT_WINDOW, legal)
        for wickets in range(1, 11):
            lost = 10 - wickets
            if lost > legal:
                continue  # more wickets down than balls bowled: not a real state
            typical = past.get((overs_left, wickets))
            if legal == 0:
                so_far, runs_30, lost_30, source = 0, 0, 0, "start"
            elif typical and typical["n"] >= MIN_PAST:
                so_far, runs_30, lost_30, source = typical["cum_runs"], typical["runs_last_30"], typical["wickets_last_30"], "median"
            else:  # too few real chases here to take a median from: par pace, wickets spread evenly
                so_far, runs_30 = round(rate * legal), round(rate * recent)
                lost_30, source = round(lost * recent / legal), "par pace"
            used[(overs_left, wickets)] = [so_far, runs_30, lost_30, typical["n"] if typical else 0, source]
            for need in range(1, max_need(overs_left) + 1):
                rows.append((overs_left, wickets, need, legal, so_far, lost, runs_30, lost_30, so_far + need))
    return rows, used


def support_levels(counts: dict, overs_left: int, wickets: int) -> str:
    """One digit per runs-needed value: how many real chases passed within NEAR_RUNS of this state."""
    out = []
    for need in range(1, max_need(overs_left) + 1):
        n = sum(counts.get((overs_left, wickets, r), 0) for r in range(need - NEAR_RUNS, need + NEAR_RUNS + 1))
        out.append(str(sum(n >= t for t in SUPPORT_LEVELS)))
    return "".join(out)


def run(spark: SparkSession, copy_to: Path | None) -> dict:
    cfg = get_settings()
    card = json.loads((cfg.models / "model_card.json").read_text())
    model = PipelineModel.load(str(cfg.models / MODEL_DIR))
    par, par_by_split = median_par(spark, cfg)

    ends = chase_over_ends(spark, cfg).persist()
    past = {(r.overs_left, r.wickets_in_hand): r.asDict() for r in
            ends.groupBy("overs_left", "wickets_in_hand")
            .agg(*[F.percentile_approx(c, 0.5).alias(c) for c in ("cum_runs", "runs_last_30", "wickets_last_30")],
                 F.count("*").alias("n")).collect()}
    counts = {(r.overs_left, r.wickets_in_hand, r.runs_needed): r["count"] for r in
              ends.groupBy("overs_left", "wickets_in_hand", "runs_needed").count().collect()}
    n_chases = ends.select("match_id").distinct().count()
    ends.unpersist()

    rows, used = chase_states(par, past)
    cols = ["overs_left", "wickets", "need", "legal_balls", "cum_runs", "cum_wickets", "runs_last_30",
            "wickets_last_30", "target_runs"]
    states = (spark.createDataFrame(rows, cols)
              .withColumns({"innings_no": F.lit(2), "target_overs": F.lit(20.0), "scheduled_overs": F.lit(20),
                            "first_innings_total": F.col("target_runs") - 1, "venue_par": F.lit(float(par)),
                            "gender": F.lit("male"), "winner": F.lit(None).cast("string"),
                            "result_type": F.lit(None).cast("string"), "batting_team": F.lit("chasing")}))
    scored = (score(model, add_state_features(states))
              .select("overs_left", "wickets", "need", "win_prob", "model_win_prob", "runs_needed", "wickets_in_hand",
                      "balls_remaining")
              .toPandas())
    # the inputs must come back as the states we asked for, and every state is unfinished, so no pinning
    assert (scored["runs_needed"] == scored["need"]).all() and (scored["wickets_in_hand"] == scored["wickets"]).all()
    assert (scored["balls_remaining"] == 6 * scored["overs_left"]).all()
    assert (scored["win_prob"] == scored["model_win_prob"]).all()

    p = {(r.overs_left, r.wickets, r.need): round(1000 * r.win_prob) for r in scored.itertuples()}
    feasible = [[(o, w) in used for w in range(1, 11)] for o in range(1, 21)]
    grid = [[[p[(o, w, n)] for n in range(1, max_need(o) + 1)] if feasible[o - 1][w - 1] else None
             for w in range(1, 11)] for o in range(1, 21)]
    support = [[support_levels(counts, o, w) if feasible[o - 1][w - 1] else None for w in range(1, 11)]
               for o in range(1, 21)]
    # how jumpy is the raw model where real data backs it, versus where it extrapolates?
    jumps = {"supported": [], "rare": []}
    for o in range(20):
        for w in range(10):
            row, sup = grid[o][w], support[o][w]
            if row:
                for i in range(len(row) - 1):
                    key = "supported" if sup[i] >= "2" and sup[i + 1] >= "2" else "rare"
                    jumps[key].append(row[i + 1] - row[i])

    test = card["metrics"][card["champion"]]["test"]
    out = {
        "what": "Win probability of the chasing side, as predicted by the trained model. p values are tenths of a "
                "percent (0-1000), indexed p[overs_left - 1][wickets_in_hand - 1][runs_needed - 1]; null where the "
                "state can't happen. support has one digit per runs-needed value: 0 = fewer than 5 real training "
                "chases within 5 runs of the state, 1 = 5-24, 2 = 25-99, 3 = 100 or more.",
        "model": {
            "name": card["model_name"], "version": card["version"], "champion": card["champion"],
            "data": "real Cricsheet T20 matches, men's and women's (conf/real.yaml)",
            "splits": card["splits"],
            "test": {k: round(test[k], 4) for k in ("log_loss", "brier_skill_score", "auc", "accuracy", "ece")},
            "test_rows": test["rows"],
        },
        "assumptions": {
            "venue_par": par, "venue_par_median_by_split": {k: round(v, 1) for k, v in par_by_split.items()},
            "past": "runs so far, runs and wickets in the last 30 balls: median of real men's training chases at "
                    f"the same over with the same wickets down; par pace ({par / 20:.2f} an over) where fewer than "
                    f"{MIN_PAST} chases exist",
            "target": "runs so far + runs needed", "gender": "male", "overs": 20,
            "training_chases": n_chases,
        },
        "checks": {
            "states": len(p),
            "one_run_steps_where_wp_rose": {k: sum(d > 0 for d in v) for k, v in jumps.items()},
            "one_run_steps": {k: len(v) for k, v in jumps.items()},
            "largest_rise_pts": {k: max(v, default=0) / 10 for k, v in jumps.items()},
        },
        "grid": {"overs_left": list(range(1, 21)), "wickets_in_hand": list(range(1, 11)),
                 "max_runs_needed": [max_need(o) for o in range(1, 21)],
                 "past": [[used.get((o, w)) for w in range(1, 11)] for o in range(1, 21)],
                 "p": grid, "support": support},
    }
    path = cfg.models / GRID_NAME
    path.write_text(json.dumps(out, separators=(",", ":")))
    log.info("%d states, par %s -> %s (%.0f KB)", len(p), par, path, path.stat().st_size / 1024)
    if copy_to:
        copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, copy_to / GRID_NAME)
    return {"states": len(p), "par": par, **out["checks"], "path": str(path)}


if __name__ == "__main__":
    from cricklake.spark import get_spark

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy-to", type=Path, default=None, help="also copy the JSON into this folder")
    print(run(get_spark("export_grid"), ap.parse_args().copy_to))
