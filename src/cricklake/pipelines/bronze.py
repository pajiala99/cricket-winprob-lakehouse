"""Bronze: packed match documents -> Delta, append-only and schema-enforced.

Runs as Structured Streaming with trigger(availableNow=True): one code path serves the historical
backfill and every later incremental run, and the checkpoint guarantees each shard is ingested exactly
once even if the job is killed and restarted.
"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cricklake.config import get_settings
from cricklake.schemas import MATCH

log = logging.getLogger("cricklake.bronze")


def run(spark: SparkSession) -> dict:
    cfg = get_settings()
    source = str(cfg.landing / "packed" / "*" / "*.ndjson")
    raw = (
        spark.readStream.schema(MATCH)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .option("maxFilesPerTrigger", 64)
        .json(source)
    )
    bronze = (
        raw.withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_batch_id", F.regexp_extract("_source_file", r"packed/([^/]+)/", 1))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    query = (
        bronze.writeStream.format("delta")
        .option("checkpointLocation", cfg.checkpoint("bronze_matches"))
        .outputMode("append")
        .trigger(availableNow=True)
        .start(cfg.table("bronze", "matches"))
    )
    query.awaitTermination()
    rows = sum(p["numInputRows"] for p in query.recentProgress)
    log.info("Bronze ingested %s match documents in %d micro-batch(es)", f"{rows:,}", len(query.recentProgress))
    return {"bronze_rows": rows}


if __name__ == "__main__":
    from cricklake.spark import get_spark

    run(get_spark("bronze"))
