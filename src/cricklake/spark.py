"""SparkSession factory with Delta Lake enabled."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from cricklake.config import get_settings

log = logging.getLogger(__name__)


def _ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    java = shutil.which("java")
    if java:
        # .../Contents/Home/bin/java -> .../Contents/Home
        os.environ["JAVA_HOME"] = str(Path(java).resolve().parent.parent)


def get_spark(app_suffix: str = "", extra_packages: list[str] | None = None) -> SparkSession:
    _ensure_java_home()
    # Python workers must run the driver's interpreter; otherwise Spark picks whatever `python3` is on PATH
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    cfg = get_settings()["spark"]
    name = cfg["app_name"] + (f"-{app_suffix}" if app_suffix else "")
    builder = (
        SparkSession.builder.appName(name)
        .master(os.environ.get("SPARK_MASTER", cfg["master"]))
        .config("spark.driver.memory", cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(cfg["shuffle_partitions"]))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
        .config("spark.databricks.delta.optimizeWrite.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.streaming.stateStore.providerClass",
                "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider")
    )
    spark = configure_spark_with_delta_pip(builder, extra_packages=extra_packages).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info("Spark %s session '%s' ready", spark.version, name)
    return spark
