"""A small declarative data-quality framework (the idea of dbt tests / Great Expectations, minus the weight).

Every check yields (failing_rows, total_rows). If the failure rate exceeds the check's tolerance it fails;
results are appended to the quality.dq_results Delta table, and any failed ERROR check raises
DataQualityError so the orchestrator stops before bad data reaches the next layer.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from cricklake.config import get_settings

log = logging.getLogger("cricklake.quality")


class DataQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Check:
    name: str
    table: str                  # "layer.table"; queries reference it as layer_table
    kind: str                   # expect | unique | foreign_key | sql
    expr: str                   # expect: TRUE for a good row | unique: key cols | foreign_key: "col -> layer.table.col"
    severity: str = "error"     # error | warn
    tolerance: float = 0.0      # accepted fraction of failing rows


SILVER_CHECKS = [
    Check("matches_pk_unique", "silver.matches", "unique", "match_id"),
    Check("matches_have_date", "silver.matches", "expect", "match_date IS NOT NULL"),
    Check("matches_distinct_teams", "silver.matches", "expect", "team_1 <> team_2"),
    Check("winner_is_a_participant", "silver.matches", "expect", "winner IS NULL OR winner IN (team_1, team_2)"),
    Check("result_type_accepted", "silver.matches", "expect",
          "result_type IN ('win', 'tie', 'no result', 'draw')"),
    Check("deliveries_pk_unique", "silver.deliveries", "unique", "match_id, innings_no, delivery_no"),
    Check("runs_add_up", "silver.deliveries", "expect", "runs_total = runs_batter + runs_extras"),
    Check("batter_runs_plausible", "silver.deliveries", "expect", "runs_batter BETWEEN 0 AND 6",
          severity="warn", tolerance=0.0005),
    Check("players_resolved_in_registry", "silver.deliveries", "expect",
          "batter_id IS NOT NULL AND bowler_id IS NOT NULL", severity="warn", tolerance=0.001),
    Check("deliveries_reference_matches", "silver.deliveries", "foreign_key", "match_id -> silver.matches.match_id"),
    Check("legal_balls_per_over", "silver.deliveries", "sql", """
        SELECT count_if(legal_balls > balls_per_over), count(*) FROM (
          SELECT match_id, innings_no, over_no, max(balls_per_over) AS balls_per_over,
                 sum(CAST(is_legal AS INT)) AS legal_balls
          FROM silver_deliveries GROUP BY match_id, innings_no, over_no)""", severity="warn", tolerance=0.01),
    Check("chase_stops_at_target", "silver.deliveries", "sql", """
        SELECT count_if(runs > target_runs + 6), count(*) FROM (
          SELECT match_id, max(target_runs) AS target_runs, sum(runs_total) AS runs
          FROM silver_deliveries WHERE innings_no = 2 AND target_runs IS NOT NULL GROUP BY match_id)"""),
]

GOLD_CHECKS = [
    Check("ball_state_pk_unique", "gold.ball_state", "unique", "match_id, innings_no, delivery_no"),
    Check("balls_remaining_in_range", "gold.ball_state", "expect", "balls_remaining BETWEEN 0 AND 120"),
    Check("wickets_in_range", "gold.ball_state", "expect", "cum_wickets BETWEEN 0 AND 10"),
    Check("label_present_for_decided", "gold.ball_state", "expect",
          "batting_team_won IN (0, 1) OR winner IS NULL OR result_type IN ('no result', 'draw')"),
    Check("win_prob_is_probability", "gold.win_probability", "expect", "win_prob BETWEEN 0 AND 1"),
    Check("wp_series_ends_at_result", "gold.win_probability", "sql", """
        SELECT count_if(final_wp NOT IN (0.0, 0.5, 1.0)), count(*) FROM (
          SELECT match_id, max_by(win_prob, delivery_no) AS final_wp, max(result_type) AS result_type
          FROM gold_win_probability WHERE innings_no = 2 GROUP BY match_id)
        WHERE result_type <> 'no result'""", severity="warn", tolerance=0.02),
    Check("player_impact_pk_unique", "gold.player_impact", "unique", "player_id, event_name, season_year"),
]


def _register(spark: SparkSession, tables: set[str]) -> None:
    cfg = get_settings()
    for t in tables:
        layer, name = t.split(".")
        spark.read.format("delta").load(cfg.table(layer, name)).createOrReplaceTempView(f"{layer}_{name}")


def _evaluate(spark: SparkSession, c: Check) -> tuple[int, int]:
    view = c.table.replace(".", "_")
    if c.kind == "expect":
        row = spark.sql(f"SELECT count_if(NOT coalesce(({c.expr}), false)), count(*) FROM {view}").first()
    elif c.kind == "unique":
        row = spark.sql(f"SELECT count(*) - count(DISTINCT {c.expr}), count(*) FROM {view}").first()
    elif c.kind == "foreign_key":
        child_col, parent = (s.strip() for s in c.expr.split("->"))
        p_layer, p_name, p_col = parent.split(".")
        row = spark.sql(f"""
            SELECT count_if(p.k IS NULL), count(*) FROM {view} c
            LEFT JOIN (SELECT DISTINCT {p_col} AS k FROM {p_layer}_{p_name}) p ON c.{child_col} = p.k""").first()
    elif c.kind == "sql":
        row = spark.sql(c.expr).first()
    else:
        raise ValueError(f"unknown check kind {c.kind}")
    return int(row[0] or 0), int(row[1] or 0)


def run_checks(spark: SparkSession, checks: list[Check], suite: str) -> list[dict]:
    cfg = get_settings()
    tables = {c.table for c in checks}
    for c in checks:
        if c.kind == "foreign_key":
            tables.add(".".join(c.expr.split("->")[1].strip().split(".")[:2]))
    _register(spark, tables)

    run_id, checked_at = uuid.uuid4().hex[:12], datetime.now(UTC)
    results = []
    for c in checks:
        failing, total = _evaluate(spark, c)
        rate = failing / total if total else 0.0
        passed = rate <= c.tolerance
        results.append({
            "run_id": run_id, "suite": suite, "check_name": c.name, "table_name": c.table, "severity": c.severity,
            "failing_rows": failing, "total_rows": total, "failure_rate": rate, "tolerance": c.tolerance,
            "passed": passed, "checked_at": checked_at,
        })
        level = logging.INFO if passed else (logging.ERROR if c.severity == "error" else logging.WARNING)
        log.log(level, "[%s] %-32s %-18s failing=%s/%s (%.4f%%)", "PASS" if passed else "FAIL", c.name, c.table,
                f"{failing:,}", f"{total:,}", 100 * rate)

    path = cfg.table("quality", "dq_results")
    df = spark.createDataFrame(results)
    df.write.format("delta").mode("append").option("mergeSchema", "true").save(path)
    if DeltaTable.isDeltaTable(spark, path):
        log.info("DQ results appended to %s (run %s)", path, run_id)

    errors = [r for r in results if not r["passed"] and r["severity"] == "error"]
    if errors:
        raise DataQualityError(f"{len(errors)} blocking check(s) failed in suite '{suite}': "
                               + ", ".join(r["check_name"] for r in errors))
    return results
