import py_compile
from pathlib import Path

import pytest

from cricklake.quality import checks
from cricklake.quality.checks import Check, DataQualityError
from cricklake.streaming.state import MatchState
from tests.conftest import make_settings


@pytest.fixture()
def dq_lake(spark, tmp_path, monkeypatch):
    cfg = make_settings(tmp_path)
    monkeypatch.setattr(checks, "get_settings", lambda: cfg)
    rows = [("1", "A", "B"), ("2", "C", "D"), ("2", "C", "C")]
    spark.createDataFrame(rows, "match_id string, team_1 string, team_2 string") \
        .write.format("delta").save(cfg.table("silver", "matches"))
    return cfg


def test_blocking_check_raises_and_results_are_recorded(spark, dq_lake):
    suite = [Check("pk", "silver.matches", "unique", "match_id"),
             Check("teams_differ", "silver.matches", "expect", "team_1 <> team_2", severity="warn")]
    with pytest.raises(DataQualityError, match="pk"):
        checks.run_checks(spark, suite, "test")
    recorded = spark.read.format("delta").load(dq_lake.table("quality", "dq_results"))
    assert {r.check_name: r.failing_rows for r in recorded.collect()} == {"pk": 1, "teams_differ": 1}


def test_warnings_and_tolerances_do_not_block(spark, dq_lake):
    suite = [Check("teams_differ", "silver.matches", "expect", "team_1 <> team_2", severity="warn"),
             Check("mostly_unique", "silver.matches", "unique", "match_id", tolerance=0.5)]
    results = checks.run_checks(spark, suite, "test")
    assert [r["passed"] for r in results] == [False, True]


def test_match_state_rolling_window_and_json_roundtrip():
    state = MatchState()
    for n in range(1, 41):
        totals = state.apply({"innings_no": 1, "delivery_no": n, "runs_total": 1, "is_wicket": n % 10 == 0,
                              "is_legal": True})
    assert totals == {"cum_runs": 40, "cum_wickets": 4, "legal_balls": 40, "runs_last_30": 30, "wickets_last_30": 3}
    restored = MatchState.from_json(state.to_json())
    assert restored.first_innings_total == 40
    assert restored.apply({"innings_no": 1, "delivery_no": 40, "runs_total": 6, "is_wicket": False,
                           "is_legal": True}) is None


def test_airflow_dag_file_compiles():
    py_compile.compile(str(Path(__file__).parents[1] / "orchestration/airflow/dags/cricket_lakehouse_dag.py"),
                       doraise=True)


def test_unique_check_counts_duplicates_with_null_keys(spark, tmp_path, monkeypatch):
    cfg = make_settings(tmp_path)
    monkeypatch.setattr(checks, "get_settings", lambda: cfg)
    rows = [("p1", None), ("p1", None), ("p2", "IPL")]  # real data: internationals without a series name
    spark.createDataFrame(rows, "player_id string, event_name string") \
        .write.format("delta").save(cfg.table("gold", "player_impact"))
    with pytest.raises(DataQualityError):
        checks.run_checks(spark, [Check("pk", "gold.player_impact", "unique", "player_id, event_name")], "test")
    recorded = spark.read.format("delta").load(cfg.table("quality", "dq_results")).first()
    assert recorded.failing_rows == 1  # exactly one duplicate, not "every row with a NULL key"
