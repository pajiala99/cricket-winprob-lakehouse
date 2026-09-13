"""Airflow DAGs for the Cricket Win-Probability Lakehouse.

cricket_lakehouse_daily           02:00 every day  - ingest the latest data drop, rebuild features, marts and
                                                     player impact with the current model, gate on data quality
cricket_lakehouse_weekly_retrain  04:00 on Sundays - retrain the model, re-score history, compact tables

Each task shells out to the same entry point used locally (`python -m cricklake.orchestrate --only <task>`),
so Airflow owns scheduling, retries and alerting while all business logic stays in the cricklake package.
"""

import os
from datetime import datetime, timedelta

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.sdk import DAG
except ImportError:  # Airflow 2
    from airflow import DAG
    from airflow.operators.bash import BashOperator

PROJECT = os.environ.get("CRICKLAKE_HOME", "/opt/cricket-winprob-lakehouse")
PYTHON = f"{PROJECT}/.venv/bin/python"

default_args = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def step(task_id: str, *flags: str) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT} && {PYTHON} -m cricklake.orchestrate --only {task_id} {' '.join(flags)}".strip(),
    )


with DAG(
    dag_id="cricket_lakehouse_daily",
    description="Incremental ingest -> silver -> features/marts -> player impact, with DQ gates",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["cricket", "lakehouse", "spark", "delta"],
    doc_md=__doc__,
):
    pack = step("pack", "--incremental")
    bronze = step("bronze")
    silver = step("silver")
    dq_silver = step("dq_silver")
    features = step("features")
    marts = step("marts")
    impact = step("impact")
    dq_gold = step("dq_gold")
    validate = step("validate_wpa")

    pack >> bronze >> silver >> dq_silver >> [features, marts]
    features >> impact >> [dq_gold, validate]
    marts >> dq_gold

with DAG(
    dag_id="cricket_lakehouse_weekly_retrain",
    description="Retrain the win-probability model, re-score all balls, compact Delta tables",
    start_date=datetime(2026, 1, 1),
    schedule="0 4 * * 0",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["cricket", "ml", "spark"],
):
    train = step("train")
    rescore = step("impact")
    dq = step("dq_gold")
    maintenance = step("maintenance")

    train >> rescore >> dq >> maintenance
