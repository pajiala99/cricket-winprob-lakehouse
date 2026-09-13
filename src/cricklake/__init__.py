"""Cricket Win-Probability Lakehouse: ball-by-ball big data pipeline, live scoring and player impact."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("py4j").setLevel(logging.WARNING)

__version__ = "1.0.0"
