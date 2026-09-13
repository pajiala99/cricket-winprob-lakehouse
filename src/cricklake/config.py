"""Configuration loading and lakehouse path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("CRICKLAKE_HOME", Path(__file__).resolve().parents[2]))
CONFIG_FILE = Path(os.environ.get("CRICKLAKE_CONFIG", PROJECT_ROOT / "conf" / "pipeline.yaml"))

LAYERS = ("bronze", "silver", "gold", "quality")


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def landing(self) -> Path:
        return self.path("landing")

    @property
    def lakehouse(self) -> Path:
        return self.path("lakehouse")

    @property
    def checkpoints(self) -> Path:
        return self.path("checkpoints")

    @property
    def models(self) -> Path:
        return self.path("models")

    @property
    def stream_inbox(self) -> Path:
        return self.path("stream_inbox")

    @property
    def runs(self) -> Path:
        return self.path("runs")

    def table(self, layer: str, name: str) -> str:
        """Filesystem location of a Delta table, e.g. table('silver', 'deliveries')."""
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}; expected one of {LAYERS}")
        return str(self.lakehouse / layer / name)

    def checkpoint(self, name: str) -> str:
        return str(self.checkpoints / name)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    with open(CONFIG_FILE) as fh:
        return Settings(yaml.safe_load(fh))
