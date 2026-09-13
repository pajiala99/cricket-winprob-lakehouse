"""Does WPA measure what it claims? Validate player impact against the simulator's latent ground truth.

Nobody knows a real cricketer's "true" skill or clutch ability, so real-world impact metrics can only be
argued about. In the simulation the truth is known, so we can measure how well each metric ranks players
(Spearman rank correlation, 1.0 = perfect ordering) and compare WPA against traditional statistics.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.stats import spearmanr

from cricklake.config import get_settings

log = logging.getLogger("cricklake.validate_wpa")
MIN_BALLS = 300
MIN_HIGH_LEVERAGE_BALLS = 40
SUMS = ["batting_wpa", "bowling_wpa", "clutch_batting_wpa", "clutch_bowling_wpa", "hl_balls_faced",
        "hl_balls_bowled", "runs", "balls_faced", "runs_conceded", "balls_bowled", "wickets"]


def _rho(a, b) -> float:
    return round(float(spearmanr(a, b).statistic), 3)


def _clutch(df: pd.DataFrame, wpa: str, clutch_wpa: str, hl_balls: str, balls: str) -> dict:
    """Clutch = WPA rate on high-leverage balls beyond what the player's WPA rate on ordinary balls predicts."""
    sub = df[(df[hl_balls] >= MIN_HIGH_LEVERAGE_BALLS) & (df[balls] > df[hl_balls])].copy()
    ordinary = (sub[wpa] - sub[clutch_wpa]) / (sub[balls] - sub[hl_balls])
    hl_rate = sub[clutch_wpa] / sub[hl_balls]
    residual = hl_rate - np.polyval(np.polyfit(ordinary, hl_rate, 1), ordinary)
    return {"players": int(len(sub)), "clutch_score_vs_true_clutch": _rho(residual, sub["clutch"]),
            "raw_high_leverage_wpa_vs_true_clutch": _rho(hl_rate, sub["clutch"])}


def run(spark: SparkSession) -> dict:
    cfg = get_settings()
    files = sorted(cfg.path("ground_truth").glob("players_*.csv"))
    if not files:
        log.warning("No simulator ground truth (real-data mode): skipping WPA validation")
        return {"skipped": True}
    truth = (pd.concat(pd.read_csv(f) for f in files)
             .groupby("player_id").agg(bat_skill=("bat_skill", "mean"), bowl_skill=("bowl_skill", "mean"),
                                       clutch=("clutch", "mean")).reset_index())
    career = (spark.read.format("delta").load(cfg.table("gold", "player_impact"))
              .groupBy("player_id").agg(*[F.sum(c).alias(c) for c in SUMS]).toPandas())
    df = career.merge(truth, on="player_id", how="inner")
    bat, bowl = df[df.balls_faced >= MIN_BALLS], df[df.balls_bowled >= MIN_BALLS]

    result = {
        "batting": {
            "players": int(len(bat)),
            "wpa_per_ball_vs_true_skill": _rho(bat.batting_wpa / bat.balls_faced, bat.bat_skill),
            "strike_rate_vs_true_skill": _rho(bat.runs / bat.balls_faced, bat.bat_skill),
        },
        "bowling": {
            "players": int(len(bowl)),
            "wpa_per_ball_vs_true_skill": _rho(bowl.bowling_wpa / bowl.balls_bowled, bowl.bowl_skill),
            "economy_vs_true_skill": _rho(-bowl.runs_conceded / bowl.balls_bowled, bowl.bowl_skill),
            "wickets_per_ball_vs_true_skill": _rho(bowl.wickets / bowl.balls_bowled, bowl.bowl_skill),
        },
        "clutch_batting": _clutch(df[df.balls_faced >= MIN_BALLS], "batting_wpa", "clutch_batting_wpa",
                                  "hl_balls_faced", "balls_faced"),
        "clutch_bowling": _clutch(df[df.balls_bowled >= MIN_BALLS], "bowling_wpa", "clutch_bowling_wpa",
                                  "hl_balls_bowled", "balls_bowled"),
    }
    cfg.models.mkdir(parents=True, exist_ok=True)
    (cfg.models / "wpa_validation.json").write_text(json.dumps(result, indent=2))
    log.info("WPA validation: %s", json.dumps(result))
    return result


if __name__ == "__main__":
    from cricklake.spark import get_spark

    run(get_spark("validate"))
