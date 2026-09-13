"""Train the ball-by-ball win-probability model with Spark MLlib.

* Time-based split (train <= 2023, validate 2024, test 2025+): a match never spans two splits and the model
  is always judged on seasons it has never seen.
* Gradient-boosted trees versus a logistic-regression baseline; the champion is picked on validation log loss.
* Metrics that matter for probabilities, not just classification: log loss, Brier (skill) score and
  expected calibration error, plus AUC and accuracy for reference.
* A model card (JSON) records metrics, calibration, feature importances, data splits and hyperparameters.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from cricklake.config import get_settings
from cricklake.pipelines.features import FEATURES, LABEL

log = logging.getLogger("cricklake.train")
EPS = 1e-6
MODEL_DIR = "win_prob"


def with_win_prob(pred: DataFrame) -> DataFrame:
    return pred.withColumn("win_prob", vector_to_array("probability").getItem(1))


def evaluate(pred: DataFrame) -> dict:
    y = F.col(LABEL).cast("double")
    p = F.least(F.greatest(F.col("win_prob"), F.lit(EPS)), F.lit(1 - EPS))
    agg = pred.select(
        F.avg(-(y * F.log(p) + (1 - y) * F.log(1 - p))).alias("log_loss"),
        F.avg((F.col("win_prob") - y) ** 2).alias("brier"),
        F.avg(((F.col("win_prob") >= 0.5).cast("double") == y).cast("double")).alias("accuracy"),
        F.avg(y).alias("base_rate"),
        F.count("*").alias("rows"),
    ).first()
    auc = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="rawPrediction").evaluate(pred)
    bins = (pred.withColumn("bin", F.least(F.floor(F.col("win_prob") * 10), F.lit(9)).cast("int"))
            .groupBy("bin").agg(F.avg("win_prob").alias("mean_predicted"), F.avg(y).alias("observed_win_rate"),
                                F.count("*").alias("n"))
            .orderBy("bin").collect())
    n = sum(b["n"] for b in bins)
    ece = sum(b["n"] / n * abs(b["mean_predicted"] - b["observed_win_rate"]) for b in bins)
    reference = agg["base_rate"] * (1 - agg["base_rate"])
    return {
        "log_loss": agg["log_loss"], "brier": agg["brier"], "brier_skill_score": 1 - agg["brier"] / reference,
        "auc": auc, "accuracy": agg["accuracy"], "ece": ece, "rows": agg["rows"],
        "calibration": [b.asDict() for b in bins],
    }


def by_progress(pred: DataFrame) -> list[dict]:
    """How sharp is the model as the match unfolds? Metrics per innings and 5-over stage."""
    y = F.col(LABEL).cast("double")
    p = F.least(F.greatest(F.col("win_prob"), F.lit(EPS)), F.lit(1 - EPS))
    rows = (pred.withColumn("stage", F.least(F.floor(F.col("legal_balls") / 30), F.lit(3)).cast("int"))
            .groupBy("innings_no", "stage")
            .agg(F.avg(-(y * F.log(p) + (1 - y) * F.log(1 - p))).alias("log_loss"),
                 F.avg((F.col("win_prob") - y) ** 2).alias("brier"),
                 F.avg(((F.col("win_prob") >= 0.5).cast("double") == y).cast("double")).alias("accuracy"),
                 F.count("*").alias("n"))
            .orderBy("innings_no", "stage").collect())
    return [{**r.asDict(), "overs": f"{r['stage'] * 5}-{r['stage'] * 5 + 5}"} for r in rows]


def run(spark: SparkSession) -> dict:
    cfg = get_settings()
    hp = cfg["model"]["gbt"]
    data = (spark.read.format("delta").load(cfg.table("gold", "ball_state"))
            .filter(F.col(LABEL).isNotNull())
            .select(*FEATURES, LABEL, "split", "match_id", "match_date")
            .persist())
    counts = {r["split"]: r["count"] for r in data.groupBy("split").count().collect()}
    date_ranges = {r["split"]: (str(r["lo"]), str(r["hi"])) for r in
                   data.groupBy("split").agg(F.min("match_date").alias("lo"), F.max("match_date").alias("hi")).collect()}
    train, valid, test = (data.filter(F.col("split") == s) for s in ("train", "valid", "test"))
    log.info("Training rows by split: %s", {k: f"{v:,}" for k, v in counts.items()})

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    candidates = {
        "gbt": Pipeline(stages=[assembler, GBTClassifier(
            labelCol=LABEL, featuresCol="features", maxIter=hp["max_iter"], maxDepth=hp["max_depth"],
            stepSize=hp["step_size"], subsamplingRate=hp["subsampling_rate"], seed=42)]),
        "logistic_regression": Pipeline(stages=[
            assembler, StandardScaler(inputCol="features", outputCol="scaled_features"),
            LogisticRegression(labelCol=LABEL, featuresCol="scaled_features", maxIter=200, regParam=1e-4)]),
    }
    fitted, report = {}, {}
    for name, pipeline in candidates.items():
        started = time.time()
        model = pipeline.fit(train)
        fitted[name] = model
        report[name] = {
            "train_seconds": round(time.time() - started, 1),
            "valid": evaluate(with_win_prob(model.transform(valid))),
            "test": evaluate(with_win_prob(model.transform(test))),
        }
        t = report[name]["test"]
        log.info("%-20s test log_loss=%.4f brier=%.4f auc=%.4f ece=%.4f (%.0fs)", name, t["log_loss"], t["brier"],
                 t["auc"], t["ece"], report[name]["train_seconds"])

    champion = min(report, key=lambda k: report[k]["valid"]["log_loss"])
    best = fitted[champion]
    cfg.models.mkdir(parents=True, exist_ok=True)
    best.write().overwrite().save(str(cfg.models / MODEL_DIR))

    last = best.stages[-1]
    weights = last.featureImportances.toArray() if champion == "gbt" else abs(last.coefficients.toArray())
    importance = sorted(({"feature": f, "importance": float(w)} for f, w in zip(FEATURES, weights, strict=True)),
                        key=lambda d: -d["importance"])
    card = {
        "model_name": "ball_by_ball_win_probability",
        "version": datetime.now(UTC).strftime("%Y%m%dT%H%M%S"),
        "champion": champion,
        "label": f"{LABEL} (1 = batting side goes on to win)",
        "features": FEATURES,
        "splits": {s: {"rows": counts.get(s, 0), "from": date_ranges.get(s, (None,))[0],
                       "to": date_ranges.get(s, (None, None))[1]} for s in ("train", "valid", "test")},
        "hyperparameters": hp,
        "metrics": report,
        "test_by_match_progress": by_progress(with_win_prob(best.transform(test))),
        "feature_importance": importance,
    }
    (cfg.models / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    with open(cfg.models / "registry.jsonl", "a") as fh:
        t = report[champion]["test"]
        fh.write(json.dumps({"version": card["version"], "champion": champion, "test_log_loss": t["log_loss"],
                             "test_brier": t["brier"], "test_auc": t["auc"], "test_ece": t["ece"]}) + "\n")
    data.unpersist()
    t = report[champion]["test"]
    log.info("Champion: %s -> saved to %s", champion, cfg.models / MODEL_DIR)
    return {"champion": champion, "test_log_loss": round(t["log_loss"], 4), "test_brier": round(t["brier"], 4),
            "test_auc": round(t["auc"], 4), "test_ece": round(t["ece"], 4)}


if __name__ == "__main__":
    from cricklake.spark import get_spark

    run(get_spark("train"))
