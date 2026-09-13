"""Live ball-by-ball feed: publishes a match one delivery at a time as events.

Two sources:
  * simulate (default): a brand-new match between the current squads of a simulated league, so the model
    has never seen it
  * --replay <match_id | latest-final>: an existing match from silver.deliveries, e.g. the latest real final

Events land in data/stream/inbox/<match_id>/ as one small JSON file per ball, written atomically
(temp file + rename), which Spark Structured Streaming consumes like a message-queue topic.

  python -m cricklake.streaming.live_feed --interval 0.5
  CRICKLAKE_CONFIG=conf/real.yaml python -m cricklake.streaming.live_feed --replay latest-final
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from datetime import UTC, date, datetime

import pandas as pd

from cricklake.config import get_settings
from cricklake.ingest import simulator as sim

log = logging.getLogger("cricklake.live_feed")


def _clean(value):
    """JSON-safe scalar: pandas/numpy NaN and numpy types become plain Python values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def load_league(league: str):
    cfg = get_settings()
    files = sorted(cfg.path("ground_truth").glob("players_*.csv"))
    if not files:
        raise SystemExit("No simulated squads here (real-data profile?). Replay a stored match instead: "
                         "--replay latest-final")
    gt = pd.concat(pd.read_csv(f) for f in files)
    gt = gt[gt.league == league]
    season = gt.season.max()
    squads = {team: [sim.Player(r.player_id, r.player_name, r.role, r.bat_skill, r.bowl_skill, r.clutch, 5)
                     for r in rows.itertuples()]
              for team, rows in gt[gt.season == season].groupby("team")}
    return season, squads


def home_ground(team: str) -> sim.Venue:
    """Most frequent home venue of the team in silver.matches (read with delta-rs, no Spark needed)."""
    from deltalake import DeltaTable

    cfg = get_settings()
    matches = DeltaTable(cfg.table("silver", "matches")).to_pandas(columns=["team_1", "venue", "city"])
    home = matches[matches.team_1 == team]
    if home.empty:
        return sim.Venue(f"{team.split()[0]} Stadium", team.split()[0], 0.0, 0.01)
    venue = home.venue.mode().iloc[0]
    return sim.Venue(venue, home[home.venue == venue].city.iloc[0], 0.0, 0.01)


def to_events(match: dict, match_id: str, league: str, season_year: int, gender: str, venue: str) -> list[dict]:
    teams = match["info"]["teams"]
    events = []
    for idx, innings in enumerate(match["innings"]):
        if innings.get("super_over"):
            continue
        batting = innings["team"]
        target = innings.get("target", {})
        delivery_no = 0
        for over in innings["overs"]:
            for d in over["deliveries"]:
                delivery_no += 1
                extras = d.get("extras", {})
                wicket = (d.get("wickets") or [{}])[0]
                events.append({
                    "event_id": f"{match_id}-{idx + 1}-{delivery_no}",
                    "match_id": match_id,
                    "event_name": league,
                    "season_year": season_year,
                    "gender": gender,
                    "venue": venue,
                    "innings_no": idx + 1,
                    "batting_team": batting,
                    "bowling_team": teams[1] if teams[0] == batting else teams[0],
                    "over_no": over["over"],
                    "delivery_no": delivery_no,
                    "batter": d["batter"],
                    "bowler": d["bowler"],
                    "runs_batter": d["runs"]["batter"],
                    "runs_extras": d["runs"]["extras"],
                    "runs_total": d["runs"]["total"],
                    "is_legal": not ("wides" in extras or "noballs" in extras),
                    "is_wicket": bool(wicket),
                    "wicket_kind": wicket.get("kind"),
                    "player_out": wicket.get("player_out"),
                    "target_runs": target.get("runs"),
                    "target_overs": target.get("overs"),
                })
    return events


def simulated_match(league: str, seed: int | None) -> tuple[str, list[dict], str]:
    rng = random.Random(seed)
    season, squads = load_league(league)
    home, away = rng.sample(sorted(squads), 2)
    year = int(str(season)[:4])
    template = next(t for t in sim.LEAGUE_TEMPLATES if league.startswith(t[0]))
    venue = home_ground(home)
    teams = [sim.Team(name, sim.slug(name), venue.city, venue, squads[name]) for name in (home, away)]
    era = -0.10 + 0.28 * (year - sim.FIRST_SEASON) / (sim.LAST_SEASON - sim.FIRST_SEASON)
    match, _, _, _ = sim.simulate_match(rng, league, template[1], str(season), year,
                                        {"name": league, "stage": "Live"}, teams[0], teams[1], date.today(), era)
    match_id = "LIVE-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    outcome = match["info"]["outcome"]
    result = outcome.get("winner") or outcome.get("eliminator") or outcome.get("result")
    return match_id, to_events(match, match_id, league, year, template[1], venue.name), \
        f"{home} vs {away} at {venue.name} (simulated; won by {result})"


def replayed_match(match_ref: str) -> tuple[str, list[dict], str]:
    """Replay a stored match (real or simulated) from silver, in ball order."""
    import pyarrow.dataset as ds
    from deltalake import DeltaTable

    cfg = get_settings()
    matches = DeltaTable(cfg.table("silver", "matches")).to_pandas(
        columns=["match_id", "match_date", "stage", "event_name", "season_year", "winner", "match_type"])
    if match_ref == "latest-final":
        finals = matches[(matches.stage == "Final") & matches.match_type.isin(["T20", "IT20"])]
        if finals.empty:
            raise SystemExit("No finals in silver.matches; pass an explicit match id")
        match_ref = finals.sort_values("match_date").iloc[-1].match_id
    info = matches[matches.match_id == match_ref]
    if info.empty:
        raise SystemExit(f"match {match_ref} not found in silver.matches")
    balls = (DeltaTable(cfg.table("silver", "deliveries")).to_pyarrow_dataset()
             .to_table(filter=ds.field("match_id") == match_ref).to_pandas())
    balls = balls[~balls.is_super_over & balls.innings_no.isin([1, 2])].sort_values(["innings_no", "delivery_no"])
    stream_id = f"REPLAY-{match_ref}-{datetime.now(UTC):%H%M%S}"
    fields = ["event_name", "season_year", "gender", "venue", "innings_no", "batting_team", "bowling_team", "over_no",
              "delivery_no", "batter", "bowler", "runs_batter", "runs_extras", "runs_total", "is_legal", "is_wicket",
              "wicket_kind", "player_out", "target_runs", "target_overs"]
    events = [{"event_id": f"{stream_id}-{int(r['innings_no'])}-{int(r['delivery_no'])}", "match_id": stream_id,
               **{f: _clean(r[f]) for f in fields}} for r in balls.to_dict("records")]
    m = info.iloc[0]
    return stream_id, events, (f"replay of {m.event_name} {m.season_year} {m.stage or ''} "
                               f"{balls.batting_team.iloc[0]} vs {balls.bowling_team.iloc[0]} (won by {m.winner})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Publish a live match ball by ball")
    ap.add_argument("--league", default="Himalayan Premier League", help="simulated league to play a new match in")
    ap.add_argument("--replay", metavar="MATCH_ID", help="replay a stored match instead; 'latest-final' picks one")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between balls (0 = burst)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = get_settings()
    match_id, events, description = replayed_match(args.replay) if args.replay \
        else simulated_match(args.league, args.seed)
    inbox = cfg.stream_inbox / match_id
    inbox.mkdir(parents=True, exist_ok=True)
    log.info("LIVE %s: %s, %d balls to publish -> %s", match_id, description, len(events), inbox)
    for event in events:
        event["event_ts"] = datetime.now(UTC).isoformat()
        name = f"{event['innings_no']}-{event['delivery_no']:03d}.json"
        tmp = inbox / f".{name}.tmp"
        tmp.write_text(json.dumps(event))
        os.replace(tmp, inbox / name)
        if args.interval:
            time.sleep(args.interval)
    log.info("Feed finished: %s", description)


if __name__ == "__main__":
    main()
