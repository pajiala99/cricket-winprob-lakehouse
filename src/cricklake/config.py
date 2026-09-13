"""Configuration loading and lakehouse path helpers.

A config file may start with `extends: <other.yaml>` and override only what differs (see conf/real.yaml).
Select a profile with CRICKLAKE_CONFIG=conf/real.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("CRICKLAKE_HOME", Path(__file__).resolve().parents[2]))
_config_env = os.environ.get("CRICKLAKE_CONFIG")
CONFIG_FILE = (Path(_config_env) if Path(_config_env).is_absolute() else PROJECT_ROOT / _config_env) \
    if _config_env else PROJECT_ROOT / "conf" / "pipeline.yaml"

LAYERS = ("bronze", "silver", "gold", "quality")


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge(merged[key], value) \
            if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    return _merge(load_config(path.parent / parent), raw) if parent else raw


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
    return Settings(load_config(CONFIG_FILE))
