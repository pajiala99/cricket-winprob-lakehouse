"""Training must not depend on how the gold table happens to be laid out (file order, core count)."""

import random

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler

from cricklake.ml.train import ROW_KEY, canonical_order


def balls(spark, order_seed: int, partitions: int):
    """The same 2,000 balls every time, in a different row order and partitioning per order_seed."""
    rng = random.Random(0)
    rows = []
    for m in range(40):
        for innings in (1, 2):
            for delivery in range(25):
                x1, x2 = rng.random(), rng.random()
                rows.append((f"m{m:02d}", innings, delivery, x1, x2, int(x1 + 0.5 * rng.random() > 0.75)))
    random.Random(order_seed).shuffle(rows)
    return spark.createDataFrame(rows, [*ROW_KEY, "x1", "x2", "y"]).repartition(partitions)


def layout(df) -> list[list[tuple]]:
    return [[tuple(r) for r in part] for part in canonical_order(df, 4).rdd.glom().collect()]


def gbt_trees(df) -> str:
    data = VectorAssembler(inputCols=["x1", "x2"], outputCol="features").transform(canonical_order(df, 4))
    model = GBTClassifier(labelCol="y", maxIter=5, maxDepth=3, subsamplingRate=0.5, seed=42).fit(data)
    return model.toDebugString.split("\n", 1)[1]   # drop the header line, which carries the random uid


def test_canonical_order_ignores_input_layout(spark):
    assert layout(balls(spark, 1, 3)) == layout(balls(spark, 2, 5))


def test_gbt_is_identical_for_reordered_input(spark):
    assert gbt_trees(balls(spark, 1, 3)) == gbt_trees(balls(spark, 2, 5))
