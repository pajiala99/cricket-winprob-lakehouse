"""Probabilistic T20 league simulator that writes Cricsheet-format match JSON.

Why simulate?
  * cricsheet.org sits behind an anti-bot page, so automated downloads fail. The pipeline still accepts
    real Cricsheet zips (drop them into data/landing/downloads/) because the output schema is identical.
  * Volume is a dial (--scale), so the Spark jobs can be exercised on millions of deliveries.
  * Every player carries latent ground-truth skills (batting, bowling, clutch). That lets us validate the
    Win-Probability-Added metric against the truth, which real data can never offer.

Realism baked in (and deliberately messy, like real feeds):
  phase-dependent scoring, scoring inflation across eras, venue & dew effects, chase pressure,
  super overs for ties, rain-reduced (D/L) chases, no-results, miscounted overs, venue-name variants,
  franchise renames, duplicate player names across leagues, and re-published match revisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from cricklake.config import get_settings

log = logging.getLogger("cricklake.simulator")

AS_OF = date(2026, 9, 13)        # nothing is generated after "today"
UPDATE_FROM = date(2026, 7, 1)   # matches from here on arrive in a later incremental drop
FIRST_SEASON, LAST_SEASON = 2012, 2026
RENAME_SEASON = 2019

LEAGUE_TEMPLATES = [
    # name, gender, split season (e.g. "2019/20")?, start month
    ("Himalayan Premier League", "male", False, 3),
    ("Deccan T20 Championship", "male", False, 4),
    ("Coastal Super Series", "male", True, 12),
    ("Northern Plains Bash", "male", True, 11),
    ("Island Cricket League", "male", False, 7),
    ("Desert Storm T20", "male", False, 2),
    ("Monsoon Women's League", "female", False, 8),
    ("Summit Women's Cup", "female", True, 12),
]

CITIES = [
    "Patiala", "Shimla", "Ludhiana", "Amritsar", "Jalandhar", "Dehradun", "Srinagar", "Jammu",
    "Hyderabad", "Pune", "Nagpur", "Nashik", "Mysuru", "Hubballi", "Vijayawada", "Warangal",
    "Kochi", "Mangaluru", "Goa", "Visakhapatnam", "Puducherry", "Kozhikode", "Udupi", "Ratnagiri",
    "Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut", "Bareilly", "Gorakhpur", "Prayagraj",
    "Port Louis", "Colombo", "Galle", "Kandy", "Hambantota", "Suva", "Apia", "Nadi",
    "Jodhpur", "Bikaner", "Jaisalmer", "Ajmer", "Udaipur", "Kota", "Bhuj", "Barmer",
    "Guwahati", "Shillong", "Bhopal", "Indore", "Raipur", "Ranchi", "Bhubaneswar", "Siliguri",
    "Nainital", "Manali", "Gangtok", "Darjeeling", "Leh", "Ooty", "Munnar", "Coorg",
]
NICKNAMES = [
    "Panthers", "Pythons", "Falcons", "Rhinos", "Tuskers", "Cobras", "Hawks", "Stallions",
    "Mavericks", "Titans", "Blazers", "Rangers", "Warriors", "Sultans", "Monarchs", "Comets",
    "Cyclones", "Thunder", "Hurricanes", "Vipers", "Wolves", "Leopards", "Eagles", "Jaguars",
]
VENUE_SUFFIX = ["Cricket Ground", "Stadium", "International Stadium", "Sports Complex", "Oval"]
MALE_FIRST = [
    "Aarav", "Kabir", "Vihaan", "Arjun", "Rohan", "Ishaan", "Dev", "Karan", "Yash", "Aditya", "Nikhil",
    "Pranav", "Samar", "Tanish", "Varun", "Zaid", "Imran", "Faisal", "Oliver", "Liam", "Noah", "Jack",
    "Ethan", "Lucas", "Kasun", "Dinesh", "Nuwan", "Tamim", "Rakib", "Jason", "Marlon", "Andre", "Tom",
    "Ben", "Sam", "Riley", "Harvey", "Rashid", "Omar", "Hamza",
]
FEMALE_FIRST = [
    "Ananya", "Diya", "Isha", "Kavya", "Meera", "Nisha", "Priya", "Riya", "Sana", "Tara", "Aisha", "Zoya",
    "Emma", "Olivia", "Ava", "Grace", "Chloe", "Lily", "Ruby", "Mia", "Nigar", "Hayley", "Deandra",
    "Sophie", "Amelia", "Harini", "Shreya", "Pooja", "Neha", "Tanvi",
]
SURNAMES = [
    "Mehra", "Thakkar", "Bajwa", "Sandhu", "Grewal", "Rathore", "Chauhan", "Iyer", "Menon", "Pillai",
    "Nair", "Reddy", "Rao", "Kulkarni", "Deshpande", "Joshi", "Bhatt", "Trivedi", "Saxena", "Mishra",
    "Tiwari", "Pandey", "Qureshi", "Siddiqui", "Ansari", "Bose", "Banerjee", "Chatterjee", "Mukherjee",
    "Sen", "Dutta", "Kaur", "Gill", "Dhillon", "Walsh", "Harris", "Cooper", "Bennett", "Fraser", "Murray",
    "Gunawardena", "Bandara", "Rahman", "Hossain", "Chowdhury", "Baptiste", "Lewis", "Grant", "Campbell",
    "Ellis", "Barker", "Carter", "Doyle", "Fleming", "Gordon",
]
SQUAD_ROLES = ["batter"] * 5 + ["keeper"] + ["allrounder"] * 3 + ["bowler"] * 5

# --- ball outcome model -------------------------------------------------------------------------------
# Outcome order: dot, 1, 2, 3, 4, 6, wicket.  Base probabilities per phase, then log-linear adjustments.
RUNS = (0, 1, 2, 3, 4, 6, None)
WICKET_IDX = 6
PHASE_PROBS = {
    "powerplay": (0.445, 0.300, 0.060, 0.005, 0.125, 0.030, 0.035),
    "middle": (0.355, 0.405, 0.080, 0.005, 0.080, 0.035, 0.040),
    "death": (0.290, 0.330, 0.090, 0.005, 0.120, 0.090, 0.075),
}
LOG_BASE = {k: tuple(math.log(p) for p in v) for k, v in PHASE_PROBS.items()}
BAT_SKILL_W = (-0.10, 0.02, 0.02, 0.0, 0.22, 0.28, -0.30)
BOWL_SKILL_W = (0.12, -0.02, -0.02, 0.0, -0.18, -0.22, 0.25)
AGGRESSION_W = (-0.25, -0.05, 0.05, 0.0, 0.30, 0.40, 0.45)
CLUTCH_BAT_W = (-0.10, 0.0, 0.0, 0.0, 0.25, 0.30, -0.35)
CLUTCH_BOWL_W = (0.10, 0.0, 0.0, 0.0, -0.25, -0.30, 0.35)
CONDITIONS_W = (-0.30, 0.0, 0.05, 0.0, 0.50, 0.60, -0.20)
WICKET_KINDS = (("caught", 0.60), ("bowled", 0.18), ("lbw", 0.12), ("run out", 0.06),
                ("stumped", 0.025), ("caught and bowled", 0.015))


@dataclass
class Player:
    pid: str
    name: str
    role: str
    bat: float
    bowl: float
    clutch: float
    career_left: int


@dataclass
class Venue:
    name: str
    city: str
    factor: float
    dew: float


@dataclass
class Team:
    name: str
    franchise: str
    city: str
    venue: Venue
    squad: list


@dataclass
class InningsResult:
    team: str
    overs: list
    runs: int = 0
    wickets: int = 0
    legal_balls: int = 0
    deliveries: int = 0
    rain_stopped: bool = False


def outcome_probs(phase, bat, bowl, aggression, conditions, clutch_bat=0.0, clutch_bowl=0.0):
    base = LOG_BASE[phase]
    logits = [
        base[i] + bat * BAT_SKILL_W[i] + bowl * BOWL_SKILL_W[i] + aggression * AGGRESSION_W[i]
        + conditions * CONDITIONS_W[i] + clutch_bat * CLUTCH_BAT_W[i] + clutch_bowl * CLUTCH_BOWL_W[i]
        for i in range(7)
    ]
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def sample(rng: random.Random, probs) -> int:
    u, acc = rng.random(), 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return len(probs) - 1


def weighted_choice(rng: random.Random, pairs):
    u, acc = rng.random(), 0.0
    for value, p in pairs:
        acc += p
        if u < acc:
            return value
    return pairs[-1][0]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def situation(res: InningsResult, target, balls_left: int, phase: str):
    """Batting intent (aggression) and whether this is a high-pressure 'clutch' moment."""
    in_hand = 10 - res.wickets
    if target is None:
        if phase == "powerplay":
            a = 0.05
        elif phase == "middle":
            a = 0.04 * (in_hand - 6)
        else:
            a = 0.20 + 0.07 * (in_hand - 5)
        return clamp(a, -0.5, 0.8), False
    needed = target - res.runs
    rrr = needed * 6 / max(balls_left, 1)
    a = (rrr - 8.0) / 4.5
    if in_hand <= 3:
        a -= 0.15 * (4 - in_hand)
    clutch = balls_left <= 36 and 6.0 <= rrr <= 14.0 and in_hand >= 2
    return clamp(a, -0.6, 1.3), clutch


def slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def new_player(rng, league_idx, serial, role, gender, used_names) -> Player:
    first = FEMALE_FIRST if gender == "female" else MALE_FIRST
    name = ""
    for _ in range(60):
        name = f"{rng.choice(first)} {rng.choice(SURNAMES)}"
        if name not in used_names:
            break
    else:
        name = f"{name} {serial}"
    used_names.add(name)
    pid = hashlib.md5(f"sim-{league_idx}-{serial}".encode()).hexdigest()[:8]
    if role == "batter":
        bat, bowl = rng.gauss(0.8, 0.45), rng.gauss(-1.5, 0.4)
    elif role == "keeper":
        bat, bowl = rng.gauss(0.5, 0.45), -2.0
    elif role == "allrounder":
        bat, bowl = rng.gauss(0.2, 0.45), rng.gauss(0.2, 0.45)
    else:
        bat, bowl = rng.gauss(-1.2, 0.35), rng.gauss(0.7, 0.45)
    return Player(pid, name, role, bat, bowl, rng.gauss(0.0, 0.6), rng.randint(5, 13))


def pick_xi(rng, team: Team) -> list:
    by_role = defaultdict(list)
    for p in team.squad:
        by_role[p.role].append(p)

    def best(role, n, key):
        return sorted(by_role[role], key=lambda p: -(key(p) + rng.gauss(0, 0.3)))[:n]

    return (best("batter", 4, lambda p: p.bat) + best("keeper", 1, lambda p: p.bat)
            + best("allrounder", 2, lambda p: p.bat + p.bowl) + best("bowler", 4, lambda p: p.bowl))


def pick_bowler(rng, options, overs_bowled, last, over, max_overs, quota):
    """Captaincy: rotate bowlers under the quota, but keep two overs of each strike bowler for the death."""
    death = over >= max_overs - 5
    reserve = 0 if death or max_overs < 10 else 2
    strike = options[:2]  # options are sorted best-first

    def allowed(p):
        cap = quota - (reserve if any(p is s for s in strike) else 0)
        return overs_bowled[p.pid] < cap and p is not last

    cands = ([p for p in options if allowed(p)]
             or [p for p in options if overs_bowled[p.pid] < quota and p is not last]
             or [p for p in options if p is not last] or options)
    return max(cands, key=lambda p: p.bowl + rng.gauss(0, 0.45) + (0.6 * p.bowl if death else 0.0))


def simulate_innings(rng, team_name, batting_xi, bowling_xi, conditions, max_overs, stats, target=None,
                     max_wickets=10, rain_stop_over=None, order=None, bowler_pool=None) -> InningsResult:
    order = order or sorted(batting_xi, key=lambda p: -(p.bat + rng.gauss(0, 0.25)))
    options = bowler_pool or sorted(bowling_xi, key=lambda p: -p.bowl)[:6]
    keeper = next((p for p in bowling_xi if p.role == "keeper"), bowling_xi[0])
    quota = max(1, math.ceil(max_overs / 5))
    overs_bowled = defaultdict(int)
    res = InningsResult(team=team_name, overs=[])
    striker, non_striker, next_in = order[0], order[1], 2
    max_balls = max_overs * 6
    last_bowler, done = None, False

    for over in range(max_overs):
        if rain_stop_over is not None and over >= rain_stop_over:
            res.rain_stopped = True
            break
        bowler = pick_bowler(rng, options, overs_bowled, last_bowler, over, max_overs, quota)
        overs_bowled[bowler.pid] += 1
        phase = "powerplay" if over < 6 else ("death" if over >= max_overs - 5 else "middle")
        balls_this_over = 7 if rng.random() < 0.002 else 6  # umpires occasionally miscount
        legal_in_over, deliveries = 0, []
        while legal_in_over < balls_this_over:
            aggression, clutch_on = situation(res, target, max_balls - res.legal_balls, phase)
            probs = outcome_probs(phase, striker.bat, bowler.bowl, aggression, conditions,
                                  striker.clutch if clutch_on else 0.0, bowler.clutch if clutch_on else 0.0)
            d = {"batter": striker.name, "bowler": bowler.name, "non_striker": non_striker.name}
            extras, runs_bat, wicket, legal = {}, 0, False, True
            u = rng.random()
            wide_p = 0.030 + (0.012 if phase == "death" else 0.0)
            if u < wide_p:
                extras["wides"] = 5 if rng.random() < 0.03 else 1
                legal = False
            elif u < wide_p + 0.005:
                extras["noballs"] = 1
                legal = False
                idx = sample(rng, probs)
                runs_bat = RUNS[idx] if idx != WICKET_IDX else 0
            else:
                idx = sample(rng, probs)
                if idx == WICKET_IDX:
                    wicket = True
                else:
                    runs_bat = RUNS[idx]
                    if runs_bat == 0 and rng.random() < 0.04:
                        extras["legbyes" if rng.random() < 0.75 else "byes"] = 1
            extra_runs = sum(extras.values())
            d["runs"] = {"batter": runs_bat, "extras": extra_runs, "total": runs_bat + extra_runs}
            if extras:
                d["extras"] = extras
            res.runs += runs_bat + extra_runs
            res.deliveries += 1
            if legal:
                res.legal_balls += 1
                legal_in_over += 1
            stats[striker.pid]["runs"] += runs_bat
            stats[bowler.pid]["conceded"] += runs_bat + extras.get("wides", 0) + extras.get("noballs", 0)

            if wicket:
                kind = weighted_choice(rng, WICKET_KINDS)
                out = striker
                w = {"player_out": "", "kind": kind}
                if kind == "run out":
                    if rng.random() < 0.35:
                        out = non_striker
                    w["fielders"] = [{"name": rng.choice(bowling_xi).name}]
                elif kind == "caught":
                    w["fielders"] = [{"name": rng.choice([p for p in bowling_xi if p is not bowler]).name}]
                elif kind == "stumped":
                    w["fielders"] = [{"name": keeper.name}]
                if kind != "run out":
                    stats[bowler.pid]["wickets"] += 1
                w["player_out"] = out.name
                d["wickets"] = [w]
                res.wickets += 1
                deliveries.append(d)
                if res.wickets >= max_wickets or next_in >= len(order):
                    done = True
                    break
                incoming = order[next_in]
                next_in += 1
                if out is striker:
                    striker = incoming
                else:
                    non_striker = incoming
            else:
                deliveries.append(d)
                if (runs_bat + extras.get("byes", 0) + extras.get("legbyes", 0)) % 2 == 1:
                    striker, non_striker = non_striker, striker
            if (target is not None and res.runs >= target) or res.legal_balls >= max_balls:
                done = True
                break
        res.overs.append({"over": over, "deliveries": deliveries})
        if done:
            break
        striker, non_striker = non_striker, striker
        last_bowler = bowler
    return res


def dl_resource(overs: int) -> float:
    return 1.0 - math.exp(-0.045 * overs)


def simulate_match(rng, league_name, gender, season_str, year, event, t1: Team, t2: Team, match_date, era):
    xi = {t1.name: pick_xi(rng, t1), t2.name: pick_xi(rng, t2)}
    venue = t1.venue if "stage" not in event else rng.choice([t1.venue, t2.venue])
    toss_winner = rng.choice([t1, t2])
    field_pref = 0.50 + 4 * venue.dew + (0.12 if year >= 2018 else 0.0)
    decision = "field" if rng.random() < field_pref else "bat"
    bat_first = toss_winner if decision == "bat" else (t2 if toss_winner is t1 else t1)
    chase = t2 if bat_first is t1 else t1
    cond1 = venue.factor + era + rng.gauss(0, 0.10)
    cond2 = cond1 + venue.dew
    stats = defaultdict(lambda: {"runs": 0, "wickets": 0, "conceded": 0})

    inn1 = simulate_innings(rng, bat_first.name, xi[bat_first.name], xi[chase.name], cond1, 20, stats)
    target_overs, rain_stop, method = 20, None, None
    r = rng.random()
    if r < 0.012:
        rain_stop = rng.randint(3, 12)
    elif r < 0.04:
        target_overs, method = rng.randint(10, 18), "D/L"
    target = inn1.runs + 1
    if method:
        target = int(round(inn1.runs * dl_resource(target_overs) / dl_resource(20))) + 1
    inn2 = simulate_innings(rng, chase.name, xi[chase.name], xi[bat_first.name], cond2, target_overs, stats,
                            target=target, rain_stop_over=rain_stop)
    innings = [
        {"team": inn1.team, "overs": inn1.overs},
        {"team": inn2.team, "overs": inn2.overs, "target": {"overs": target_overs, "runs": target}},
    ]
    deliveries = inn1.deliveries + inn2.deliveries

    if inn2.rain_stopped and inn2.runs < target:
        outcome, winner = {"result": "no result"}, None
    elif inn2.runs >= target:
        outcome, winner = {"winner": chase.name, "by": {"wickets": 10 - inn2.wickets}}, chase
    elif inn2.runs == target - 1:
        # tie -> super over: chasing side bats first, top three batters, best bowler
        so_order = {t.name: sorted(xi[t.name], key=lambda p: -p.bat)[:3] for t in (t1, t2)}
        best_bowler = {t.name: sorted(xi[t.name], key=lambda p: -p.bowl)[:1] for t in (t1, t2)}
        so1 = simulate_innings(rng, chase.name, xi[chase.name], xi[bat_first.name], cond2, 1, stats,
                               max_wickets=2, order=so_order[chase.name], bowler_pool=best_bowler[bat_first.name])
        so2 = simulate_innings(rng, bat_first.name, xi[bat_first.name], xi[chase.name], cond2, 1, stats,
                               target=so1.runs + 1, max_wickets=2, order=so_order[bat_first.name],
                               bowler_pool=best_bowler[chase.name])
        innings += [{"team": so1.team, "overs": so1.overs, "super_over": True},
                    {"team": so2.team, "overs": so2.overs, "super_over": True}]
        deliveries += so1.deliveries + so2.deliveries
        if so2.runs > so1.runs:
            winner = bat_first
        elif so2.runs < so1.runs:
            winner = chase
        else:
            winner = rng.choice([t1, t2])
        outcome = {"result": "tie", "eliminator": winner.name}
    else:
        outcome, winner = {"winner": bat_first.name, "by": {"runs": target - 1 - inn2.runs}}, bat_first
    if method:
        outcome["method"] = method

    everyone = xi[t1.name] + xi[t2.name]
    info = {
        "balls_per_over": 6,
        "city": venue.city,
        "dates": [match_date.isoformat()],
        "event": event,
        "gender": gender,
        "match_type": "T20",
        "outcome": outcome,
        "overs": 20,
        "players": {t1.name: [p.name for p in xi[t1.name]], t2.name: [p.name for p in xi[t2.name]]},
        "registry": {"people": {p.name: p.pid for p in everyone}},
        "season": season_str,
        "team_type": "club",
        "teams": [t1.name, t2.name],
        "toss": {"decision": decision, "winner": toss_winner.name},
        "venue": f"{venue.name}, {venue.city}" if rng.random() < 0.35 else venue.name,
    }
    if winner is not None:
        potm = max(xi[winner.name], key=lambda p: stats[p.pid]["runs"] + 25 * stats[p.pid]["wickets"])
        info["player_of_match"] = [potm.name]
    match = {"meta": {"data_version": "1.1.0", "created": match_date.isoformat(), "revision": 1},
             "info": info, "innings": innings}
    return match, winner, deliveries, venue


def evolve_squads(rng, teams, league_idx, serial, gender, used_names):
    """Off-season: players age, retire and get replaced; an 'auction' moves players between franchises."""
    for team in teams:
        for i, p in enumerate(team.squad):
            p.career_left -= 1
            p.bat += rng.gauss(0, 0.08)
            p.bowl += rng.gauss(0, 0.08)
            p.clutch += rng.gauss(0, 0.05)
            if p.career_left <= 0:
                team.squad[i] = new_player(rng, league_idx, next(serial), p.role, gender, used_names)
    for _ in range(len(teams) * 2):
        a, b = rng.sample(teams, 2)
        i = rng.randrange(len(SQUAD_ROLES))  # same slot index => same role, squad balance preserved
        a.squad[i], b.squad[i] = b.squad[i], a.squad[i]


def simulate_league(job) -> dict:
    league_idx, name, gender, split, start_month, seed, out_root, gt_root = job
    rng = random.Random(seed * 1_000 + league_idx)
    out_root, gt_root = Path(out_root), Path(gt_root)
    used_names: set[str] = set()
    serial = itertools.count()

    teams = []
    for k in range(8):
        city = CITIES[(league_idx * 8 + k) % len(CITIES)]
        nickname = NICKNAMES[(league_idx * 5 + k) % len(NICKNAMES)]
        venue = Venue(f"{city} {VENUE_SUFFIX[(league_idx + k) % len(VENUE_SUFFIX)]}", city,
                      rng.gauss(0, 0.12), max(0.0, rng.gauss(0.01, 0.02)))
        squad = [new_player(rng, league_idx, next(serial), role, gender, used_names) for role in SQUAD_ROLES]
        team_name = f"{city} {nickname}"
        teams.append(Team(team_name, slug(f"{name}-{team_name}"), city, venue, squad))
    aliases = [(name, t.franchise, t.name, FIRST_SEASON) for t in teams]

    tag = f"league_{league_idx:02d}"
    z_initial = zipfile.ZipFile(out_root / "initial" / f"{tag}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=5)
    z_update = zipfile.ZipFile(out_root / "update" / f"{tag}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=5)
    gt_rows, n_matches, n_deliveries, n_revised = [], 0, 0, 0

    for s_idx, year in enumerate(range(FIRST_SEASON, LAST_SEASON + 1)):
        if s_idx:
            evolve_squads(rng, teams, league_idx, serial, gender, used_names)
        if year == RENAME_SEASON:
            renamed = teams[2]
            renamed.name = f"{renamed.city} United"
            aliases.append((name, renamed.franchise, renamed.name, year))
        season_str = f"{year}/{(year + 1) % 100:02d}" if split else str(year)
        era = -0.10 + 0.28 * (year - FIRST_SEASON) / (LAST_SEASON - FIRST_SEASON)
        start = date(year, start_month, 1) + timedelta(days=rng.randint(0, 9))
        for t in teams:
            gt_rows += [(name, season_str, t.name, p.pid, p.name, p.role, round(p.bat, 4), round(p.bowl, 4),
                         round(p.clutch, 4)) for p in t.squad]

        fixtures = [(a, b, None) for a in teams for b in teams if a is not b]
        rng.shuffle(fixtures)
        wins: dict[str, int] = defaultdict(int)
        results: dict[str, Team | None] = {}
        match_no = 0
        season_complete = True

        def play(t1, t2, stage, start=start, season_str=season_str, year=year, era=era, s_idx=s_idx):
            nonlocal match_no, n_matches, n_deliveries, n_revised
            match_no += 1
            match_date = start + timedelta(days=int(match_no * 0.85))
            if match_date > AS_OF:
                return None, False
            event = {"name": name, "match_number": match_no} if stage is None else {"name": name, "stage": stage}
            match, winner, deliveries, venue = simulate_match(rng, name, gender, season_str, year, event,
                                                              t1, t2, match_date, era)
            match_id = str(5_000_000 + league_idx * 10_000 + s_idx * 100 + match_no)
            target = z_update if match_date >= UPDATE_FROM else z_initial
            target.writestr(f"{match_id}.json", json.dumps(match, separators=(",", ":")))
            if target is z_initial and rng.random() < 0.015:
                # a later correction: canonical venue name, bumped revision
                match["meta"]["revision"] = 2
                match["info"]["venue"] = venue.name
                z_update.writestr(f"{match_id}.json", json.dumps(match, separators=(",", ":")))
                n_revised += 1
            n_matches += 1
            n_deliveries += deliveries
            return winner, True

        for t1, t2, _ in fixtures:
            winner, played = play(t1, t2, None)
            if not played:
                season_complete = False
                break
            if winner:
                wins[winner.name] += 1
        if not season_complete:
            continue
        top4 = sorted(teams, key=lambda t: (-wins[t.name], rng.random()))[:4]

        def decided(w, a, b):
            return w if w is not None else rng.choice([a, b])

        q1, ok = play(top4[0], top4[1], "Qualifier 1")
        if not ok:
            continue
        q1 = decided(q1, top4[0], top4[1])
        q1_loser = top4[1] if q1 is top4[0] else top4[0]
        el, ok = play(top4[2], top4[3], "Eliminator")
        if not ok:
            continue
        el = decided(el, top4[2], top4[3])
        q2, ok = play(q1_loser, el, "Qualifier 2")
        if not ok:
            continue
        q2 = decided(q2, q1_loser, el)
        results["final"], _ = play(q1, q2, "Final")

    # real feeds occasionally ship a truncated file; the packer must reject it without failing the run
    z_initial.writestr(f"{5_900_000 + league_idx}.json", '{"meta":{"data_version":"1.1.0"},"info":{"dates":["2020-0')
    z_initial.close()
    z_update.close()
    with open(gt_root / f"players_{tag}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["league", "season", "team", "player_id", "player_name", "role",
                         "bat_skill", "bowl_skill", "clutch"])
        writer.writerows(gt_rows)
    return {"league": name, "matches": n_matches, "deliveries": n_deliveries, "revised": n_revised,
            "aliases": aliases}


def main(argv=None) -> None:
    cfg = get_settings()
    sim = cfg["simulation"]
    ap = argparse.ArgumentParser(description="Simulate T20 leagues in Cricsheet JSON format")
    ap.add_argument("--scale", type=int, default=sim["scale"], help="league multiplier (1 = 8 leagues)")
    ap.add_argument("--seed", type=int, default=sim["seed"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)

    out_root = cfg.landing / "downloads" / "synthetic"
    gt_root = cfg.path("ground_truth")
    for d in (out_root / "initial", out_root / "update", gt_root, cfg.landing / "reference"):
        d.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i in range(8 * args.scale):
        name, gender, split, month = LEAGUE_TEMPLATES[i % len(LEAGUE_TEMPLATES)]
        if i >= len(LEAGUE_TEMPLATES):
            name = f"{name} Division {i // len(LEAGUE_TEMPLATES) + 1}"
        jobs.append((i, name, gender, split, month, args.seed, str(out_root), str(gt_root)))

    log.info("Simulating %d leagues x %d seasons on %d workers", len(jobs),
             LAST_SEASON - FIRST_SEASON + 1, args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(simulate_league, jobs))

    with open(cfg.landing / "reference" / "team_aliases.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["event_name", "franchise_id", "team_name", "valid_from_season"])
        for r in results:
            writer.writerows(r["aliases"])
    total_m = sum(r["matches"] for r in results)
    total_d = sum(r["deliveries"] for r in results)
    log.info("Done: %s matches, %s deliveries, %s revised matches -> %s",
             f"{total_m:,}", f"{total_d:,}", sum(r["revised"] for r in results), out_root)
    return {"leagues": len(jobs), "matches": total_m, "deliveries": total_d}


if __name__ == "__main__":
    main()
