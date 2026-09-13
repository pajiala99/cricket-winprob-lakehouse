"""Incremental per-match running totals for the live scorer.

Pure Python (no Spark), so the same logic runs inside Spark's applyInPandasWithState and in unit tests.
Model features are then derived from these totals with cricklake.pipelines.features.add_state_features,
the exact function the batch pipeline uses, so offline training and online scoring cannot drift apart.
A unit test checks these running totals against the batch window-function totals for whole matches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

RECENT_WINDOW = 30


@dataclass
class InningsState:
    cum_runs: int = 0
    cum_wickets: int = 0
    legal_balls: int = 0
    last_delivery_no: int = 0
    recent_runs: list = field(default_factory=list)
    recent_wickets: list = field(default_factory=list)


@dataclass
class MatchState:
    innings: dict = field(default_factory=dict)  # "1" / "2" -> InningsState

    def to_json(self) -> str:
        return json.dumps({k: asdict(v) for k, v in self.innings.items()})

    @classmethod
    def from_json(cls, raw: str | None) -> MatchState:
        if not raw:
            return cls()
        return cls({k: InningsState(**v) for k, v in json.loads(raw).items()})

    @property
    def first_innings_total(self) -> int | None:
        first = self.innings.get("1")
        return first.cum_runs if first else None

    def apply(self, ball: dict) -> dict | None:
        """Fold one ball into the state. Returns running totals, or None for a duplicate / stale ball."""
        inn = self.innings.setdefault(str(int(ball["innings_no"])), InningsState())
        if int(ball["delivery_no"]) <= inn.last_delivery_no:
            return None  # at-least-once delivery: a replayed event must not be counted twice
        wicket = int(bool(ball["is_wicket"]))
        inn.cum_runs += int(ball["runs_total"])
        inn.cum_wickets += wicket
        inn.legal_balls += int(bool(ball["is_legal"]))
        inn.recent_runs = (inn.recent_runs + [int(ball["runs_total"])])[-RECENT_WINDOW:]
        inn.recent_wickets = (inn.recent_wickets + [wicket])[-RECENT_WINDOW:]
        inn.last_delivery_no = int(ball["delivery_no"])
        return {
            "cum_runs": inn.cum_runs,
            "cum_wickets": min(inn.cum_wickets, 10),
            "legal_balls": inn.legal_balls,
            "runs_last_30": sum(inn.recent_runs),
            "wickets_last_30": sum(inn.recent_wickets),
        }
