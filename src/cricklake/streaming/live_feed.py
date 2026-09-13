"""Live ball-by-ball feed: plays a brand-new match and publishes every delivery as an event.

Events land in data/stream/inbox/<match_id>/ as one small JSON file per ball, written atomically
(temp file + rename), which Spark Structured Streaming consumes like a message-queue topic.
The match is contested by the current squads of a simulated league (with their latent skills) at the home
ground of the first team, so the model has never seen this match.

  python -m cricklake.streaming.live_feed --interval 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import UTC, date, datetime

import pandas as pd

from cricklake.config import get_settings
from cricklake.ingest import simulator as sim

log = logging.getLogger("cricklake.live_feed")


def load_league(league: str):
    cfg = get_settings()
    files = sorted(cfg.path("ground_truth").glob("players_*.csv"))
    if not files:
        raise SystemExit("No simulated squads found: run `python -m cricklake.orchestrate` first")
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Publish a simulated live match ball by ball")
    ap.add_argument("--league", default="Himalayan Premier League")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between balls (0 = burst)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = get_settings()
    rng = random.Random(args.seed)
    season, squads = load_league(args.league)
    home, away = rng.sample(sorted(squads), 2)
    year = int(str(season)[:4])
    template = next(t for t in sim.LEAGUE_TEMPLATES if args.league.startswith(t[0]))
    venue = home_ground(home)
    teams = [sim.Team(name, sim.slug(name), venue.city, venue, squads[name]) for name in (home, away)]
    era = -0.10 + 0.28 * (year - sim.FIRST_SEASON) / (sim.LAST_SEASON - sim.FIRST_SEASON)
    match, winner, _, _ = sim.simulate_match(rng, args.league, template[1], str(season), year,
                                             {"name": args.league, "stage": "Live"}, teams[0], teams[1],
                                             date.today(), era)
    match_id = "LIVE-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    events = to_events(match, match_id, args.league, year, template[1], venue.name)

    inbox = cfg.stream_inbox / match_id
    inbox.mkdir(parents=True, exist_ok=True)
    log.info("LIVE %s: %s vs %s at %s, %d balls to publish -> %s", match_id, home, away, venue.name, len(events), inbox)
    for event in events:
        event["event_ts"] = datetime.now(UTC).isoformat()
        name = f"{event['innings_no']}-{event['delivery_no']:03d}.json"
        tmp = inbox / f".{name}.tmp"
        tmp.write_text(json.dumps(event))
        os.replace(tmp, inbox / name)
        if args.interval:
            time.sleep(args.interval)
    outcome = match["info"]["outcome"]
    log.info("Match over: %s", outcome.get("winner") or outcome.get("eliminator") or outcome.get("result"))


if __name__ == "__main__":
    main()
