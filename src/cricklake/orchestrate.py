"""Local orchestrator: runs the lakehouse as a DAG of tasks with retries, timings and a run report.

The Airflow DAGs in orchestration/airflow/dags/ schedule the very same tasks (one `--only <task>` per task).

Examples
  python -m cricklake.orchestrate --fresh               # clean full build: simulate -> ... -> maintenance
  python -m cricklake.orchestrate --incremental         # process the next data drop, reuse the trained model
  python -m cricklake.orchestrate --only train impact   # run selected tasks
  python -m cricklake.orchestrate --from features       # resume from a task
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cricklake.config import get_settings
from cricklake.quality.checks import DataQualityError

log = logging.getLogger("cricklake.orchestrate")


class Context:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._spark = None

    @property
    def spark(self):
        if self._spark is None:
            from cricklake.spark import get_spark

            self._spark = get_spark("pipeline")
        return self._spark


def t_simulate(ctx: Context) -> dict:
    from cricklake.ingest import simulator

    return simulator.main(["--scale", str(ctx.args.scale)] if ctx.args.scale else [])


def t_pack(ctx: Context) -> dict:
    from cricklake.ingest.pack import pack

    default = "synthetic/update/*.zip" if ctx.args.incremental else "synthetic/initial/*.zip"
    return pack(ctx.args.pattern or default)


def t_bronze(ctx: Context) -> dict:
    from cricklake.pipelines import bronze

    return bronze.run(ctx.spark)


def t_silver(ctx: Context) -> dict:
    from cricklake.pipelines import silver

    return silver.run(ctx.spark)


def _dq(ctx: Context, suite: str) -> dict:
    from cricklake.quality.checks import GOLD_CHECKS, SILVER_CHECKS, run_checks

    results = run_checks(ctx.spark, SILVER_CHECKS if suite == "silver" else GOLD_CHECKS, suite)
    return {"checks": len(results), "passed": sum(r["passed"] for r in results),
            "warnings": [r["check_name"] for r in results if not r["passed"]]}


def t_features(ctx: Context) -> dict:
    from cricklake.pipelines import features

    return features.run(ctx.spark)


def t_marts(ctx: Context) -> dict:
    from cricklake.pipelines import marts

    return marts.run(ctx.spark)


def t_train(ctx: Context) -> dict:
    from cricklake.ml import train

    return train.run(ctx.spark)


def t_impact(ctx: Context) -> dict:
    from cricklake.pipelines import impact

    return impact.run(ctx.spark)


def t_validate_wpa(ctx: Context) -> dict:
    from cricklake.ml import validate_wpa

    return validate_wpa.run(ctx.spark)


def t_maintenance(ctx: Context) -> dict:
    """Compact small files and Z-order the biggest tables by match_id for fast per-match lookups."""
    from delta.tables import DeltaTable

    cfg = get_settings()
    out = {}
    for layer, name in (("silver", "deliveries"), ("gold", "win_probability")):
        metrics = (DeltaTable.forPath(ctx.spark, cfg.table(layer, name)).optimize().executeZOrderBy("match_id")
                   .select("metrics.numFilesAdded", "metrics.numFilesRemoved").first())
        out[f"{layer}.{name}"] = {"files_removed": metrics["numFilesRemoved"], "files_added": metrics["numFilesAdded"]}
    return out


@dataclass(frozen=True)
class Task:
    name: str
    fn: Callable[[Context], Any]
    retries: int = 1


TASKS = [
    Task("simulate", t_simulate, retries=0),
    Task("pack", t_pack),
    Task("bronze", t_bronze),
    Task("silver", t_silver),
    Task("dq_silver", lambda c: _dq(c, "silver"), retries=0),
    Task("features", t_features),
    Task("marts", t_marts),
    Task("train", t_train),
    Task("impact", t_impact),
    Task("dq_gold", lambda c: _dq(c, "gold"), retries=0),
    Task("validate_wpa", t_validate_wpa),
    Task("maintenance", t_maintenance),
]
BY_NAME = {t.name: t for t in TASKS}
INCREMENTAL = ["pack", "bronze", "silver", "dq_silver", "features", "marts", "impact", "dq_gold", "validate_wpa"]


def wipe() -> None:
    cfg = get_settings()
    for path in (cfg.lakehouse, cfg.checkpoints, cfg.landing / "packed", cfg.landing / "rejected", cfg.models,
                 cfg.stream_inbox.parent):
        shutil.rmtree(path, ignore_errors=True)
    log.info("Wiped lakehouse, checkpoints, packed shards, models and stream data")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the cricket lakehouse pipeline")
    ap.add_argument("--fresh", action="store_true", help="wipe derived data first (keeps downloads)")
    ap.add_argument("--incremental", action="store_true", help="process the next data drop with the current model")
    ap.add_argument("--only", nargs="+", choices=list(BY_NAME), help="run exactly these tasks")
    ap.add_argument("--from", dest="start", choices=list(BY_NAME), help="start from this task")
    ap.add_argument("--skip", nargs="+", default=[], choices=list(BY_NAME))
    ap.add_argument("--pattern", help="zip glob under landing/downloads for the pack task")
    ap.add_argument("--scale", type=int, help="simulator league multiplier")
    args = ap.parse_args(argv)

    if args.only:
        selected = args.only
    else:
        selected = INCREMENTAL if args.incremental else [t.name for t in TASKS]
        if args.start:
            selected = selected[selected.index(args.start):]
    selected = [s for s in selected if s not in args.skip]

    if args.fresh:
        wipe()
    cfg = get_settings()
    ctx = Context(args)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    report = {"run_id": run_id, "mode": "incremental" if args.incremental else "full", "tasks": [],
              "started_at": datetime.now(UTC).isoformat()}
    failed = False
    started = time.time()
    for name in selected:
        task = BY_NAME[name]
        if failed:
            report["tasks"].append({"name": name, "status": "skipped"})
            continue
        for attempt in range(1, task.retries + 2):
            t0 = time.time()
            try:
                log.info("▶ %s (attempt %d)", name, attempt)
                output = task.fn(ctx) or {}
                secs = round(time.time() - t0, 1)
                report["tasks"].append({"name": name, "status": "success", "attempts": attempt, "seconds": secs,
                                        "output": output})
                log.info("✔ %s finished in %.1fs", name, secs)
                break
            except Exception as exc:  # noqa: BLE001 - the orchestrator must record every failure
                secs = round(time.time() - t0, 1)
                log.exception("✘ %s failed after %.1fs", name, secs)
                if isinstance(exc, DataQualityError) or attempt > task.retries:
                    report["tasks"].append({"name": name, "status": "failed", "attempts": attempt, "seconds": secs,
                                            "error": f"{type(exc).__name__}: {exc}"[:2000]})
                    failed = True
                    break
                time.sleep(3 * attempt)

    report.update(status="failed" if failed else "success", total_seconds=round(time.time() - started, 1),
                  finished_at=datetime.now(UTC).isoformat())
    cfg.runs.mkdir(parents=True, exist_ok=True)
    (cfg.runs / f"{run_id}.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\nRun {run_id} [{report['status'].upper()}] in {report['total_seconds']}s")
    for t in report["tasks"]:
        print(f"  {t['status']:<8} {t['name']:<13} {t.get('seconds', '-'):>7}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
