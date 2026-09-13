"""Streamlit dashboard for the Cricket Win-Probability Lakehouse.

Serves the gold Delta tables directly with delta-rs (no Spark cluster needed to read analytics).
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pyarrow.dataset as ds
import streamlit as st
from deltalake import DeltaTable

from cricklake.config import get_settings

st.set_page_config(page_title="Cricket Win-Probability Lakehouse", page_icon="🏏", layout="wide")
CFG = get_settings()

# Reference data-viz palette (validated): categorical slots in fixed order, chrome and status tokens per mode.
PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "ink": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70", "#c98500"], "ink": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "surface": "#1a1a19"},
}
STATUS = {"pass": ("✅", "Pass"), "warn": ("⚠️", "Warning"), "fail": ("⛔", "Failed")}


def _mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit: no theme context
        return "light"


T = PALETTE[_mode()]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


# ---------------------------------------------------------------------------------------------- data access
def _exists(layer: str, name: str) -> bool:
    """A Delta table exists once its first commit file is written (the log folder alone can be mid-creation)."""
    log_dir = Path(CFG.table(layer, name)) / "_delta_log"
    return log_dir.exists() and any(log_dir.glob("*.json"))


@st.cache_data(ttl=120, show_spinner=False)
def table(layer: str, name: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    if not _exists(layer, name):
        return pd.DataFrame()
    return DeltaTable(CFG.table(layer, name)).to_pandas(columns=list(columns) if columns else None)


@st.cache_data(ttl=120, show_spinner=False)
def match_balls(match_id: str) -> pd.DataFrame:
    dataset = DeltaTable(CFG.table("gold", "win_probability")).to_pyarrow_dataset()
    balls = dataset.to_table(filter=ds.field("match_id") == match_id).to_pandas()
    return balls.sort_values(["innings_no", "delivery_no"]).reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def read_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def runs_history() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(CFG.runs.glob("*.json"), reverse=True)]


# ---------------------------------------------------------------------------------------------- chart helpers
def base_layout(fig: go.Figure, title: str, height: int = 360, y_title: str | None = None,
                x_title: str | None = None) -> go.Figure:
    axis = dict(gridcolor=T["grid"], linecolor=T["axis"], zeroline=False, tickfont=dict(color=T["muted"]),
                title_font=dict(color=T["secondary"]), showline=True)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=T["ink"]), x=0, xanchor="left"),
        height=height, margin=dict(l=8, r=16, t=48, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=T["secondary"], size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=T["secondary"])),
        hoverlabel=dict(font_family=FONT), barcornerradius=4,
    )
    fig.update_xaxes(**axis, title_text=x_title)
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def show(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def show_table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def hbar(df: pd.DataFrame, value: str, label: str, title: str, fmt: str = "{:+.2f}", height: int | None = None):
    df = df.sort_values(value)
    fig = go.Figure(go.Bar(
        x=df[value], y=df[label], orientation="h", marker_color=T["series"][0], width=0.6,
        text=[fmt.format(v) for v in df[value]], textposition="outside", textfont=dict(color=T["secondary"]),
        hovertemplate="%{y}<br>%{x:.3f}<extra></extra>", cliponaxis=False))
    return base_layout(fig, title, height=height or max(260, 26 * len(df) + 80))


def overs(legal_balls: int) -> str:
    return f"{int(legal_balls) // 6}.{int(legal_balls) % 6}"


def worm(balls: pd.DataFrame, title: str, n_labels: int = 3) -> go.Figure:
    """Win probability of the side batting first, ball by ball across both innings."""
    first = balls.loc[balls.innings_no == 1, "batting_team"]
    team = first.iloc[0] if len(first) else balls.batting_team.iloc[0]
    rival = balls.bowling_team[balls.batting_team == team].iloc[0] if (balls.batting_team == team).any() \
        else balls.batting_team.iloc[0]
    b = balls.copy()
    b["wp_team"] = b.win_prob.where(b.batting_team == team, 1 - b.win_prob)
    b["x"] = range(1, len(b) + 1)
    b["hover"] = [
        f"Innings {r.innings_no} · over {overs(r.legal_balls)}<br>{r.batting_team} {r.cum_runs}/{r.cum_wickets}"
        f"<br>{r.batter} v {r.bowler}: {'WICKET' if r.is_wicket else str(r.runs_total) + ' run(s)'}"
        f"<br>{team} win probability {100 * r.wp_team:.1f}%" for r in b.itertuples()]
    fig = go.Figure()
    fig.add_hline(y=50, line_width=1, line_color=T["axis"])
    if (b.innings_no == 2).any():
        fig.add_vline(x=b.loc[b.innings_no == 2, "x"].iloc[0] - 0.5, line_width=1, line_color=T["axis"])
    fig.add_trace(go.Scatter(x=b.x, y=100 * b.wp_team, mode="lines", line=dict(width=2, color=T["series"][0]),
                             name=f"{team} win probability", text=b.hover, hovertemplate="%{text}<extra></extra>"))
    swings = b.reindex(b.delta_wp.abs().sort_values(ascending=False).index).head(n_labels) \
        if "delta_wp" in b else b.head(0)
    fig.add_trace(go.Scatter(
        x=swings.x, y=100 * swings.wp_team, mode="markers+text", showlegend=False,
        marker=dict(size=10, color=T["series"][0], line=dict(width=2, color=T["surface"])),
        text=[(("W " if r.is_wicket else f"{r.runs_total} ") + f"{100 * r.delta_wp:+.0f}pp") if i == 0 else ""
              for i, r in enumerate(swings.itertuples())],  # label only the biggest swing; hover carries the rest
        textposition="top center", textfont=dict(color=T["secondary"], size=11), hoverinfo="skip"))
    fig = base_layout(fig, title, height=380, y_title=f"{team} win probability (%)", x_title="Ball")
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.add_annotation(x=0, y=103, xref="paper", yref="y", showarrow=False, xanchor="left",
                       text=f"above 50%: {team} favoured · below: {rival} favoured", font=dict(color=T["muted"], size=11))
    return fig


# ---------------------------------------------------------------------------------------------- page
st.title("🏏 Cricket Win-Probability Lakehouse")
st.caption("Ball-by-ball big data pipeline · Spark 4 · Delta Lake · Structured Streaming · MLlib · "
           "data: simulated leagues in Cricsheet format")

excitement = table("gold", "match_excitement")
if excitement.empty:
    st.warning("The lakehouse is empty. Run `make pipeline` first.")
    st.stop()

card = read_json(str(CFG.models / "model_card.json"))
matches = table("silver", "matches", ("match_id",))
players = table("silver", "players", ("player_id",))
wp_rows = DeltaTable(CFG.table("gold", "win_probability")).to_pyarrow_dataset().count_rows()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Matches", f"{len(matches):,}")
k2.metric("Balls scored", f"{wp_rows:,}")
k3.metric("Players", f"{len(players):,}")
if card:
    test = card["metrics"][card["champion"]]["test"]
    k4.metric("Model AUC (unseen seasons)", f"{test['auc']:.3f}")
    k5.metric("Brier skill score", f"{test['brier_skill_score']:.1%}")

# one filter row scoping every tab
f1, f2 = st.columns([2, 1])
leagues = sorted(excitement.event_name.unique())
league = f1.selectbox("Competition", leagues, index=0)
seasons = sorted(excitement.loc[excitement.event_name == league, "season_year"].unique(), reverse=True)
season = f2.selectbox("Season", seasons, index=0)

tabs = st.tabs(["Match centre", "Player impact", "Live", "League & venues", "Model", "Data quality & runs"])

with tabs[0]:
    pool = excitement[(excitement.event_name == league) & (excitement.season_year == season)] \
        .sort_values("excitement_index", ascending=False)
    labels = {r.match_id: f"{pd.Timestamp(r.match_date):%d %b %Y} · {r.batting_first} v {r.chasing} · "
                          f"excitement {r.excitement_index:.2f}{' · great escape' if r.is_great_escape else ''}"
              for r in pool.itertuples()}
    match_id = st.selectbox("Match (most thrilling first)", list(labels), format_func=labels.get)
    balls = match_balls(match_id)
    row = pool[pool.match_id == match_id].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winner", row.winner or row.result_type)
    c2.metric("Excitement index (Σ|ΔWP|)", f"{row.excitement_index:.2f}")
    c3.metric("Biggest single-ball swing", f"{100 * row.biggest_swing:.1f} pp")
    c4.metric("Winner's lowest win probability", f"{100 * row.winner_lowest_wp:.1f}%")
    show(worm(balls, "Win probability, ball by ball"))
    moments = balls.reindex(balls.delta_wp.abs().sort_values(ascending=False).index).head(8)
    st.markdown("**Key moments**")
    show_table(pd.DataFrame({
        "Innings": moments.innings_no, "Over": moments.legal_balls.map(overs), "Batting": moments.batting_team,
        "Batter": moments.batter, "Bowler": moments.bowler,
        "Ball": ["WICKET" if w else f"{r} run(s)" for w, r in zip(moments.is_wicket, moments.runs_total, strict=True)],
        "ΔWP (batting side)": (100 * moments.delta_wp).round(1).astype(str) + " pp",
        "Leverage": moments.leverage_index.round(2)}))

with tabs[1]:
    impact = table("gold", "player_impact")
    season_impact = impact[(impact.event_name == league) & (impact.season_year == season)]
    left, right = st.columns(2)
    with left:
        top = season_impact.nlargest(12, "total_wpa").assign(label=lambda d: d.player_name + " · " + d.team)
        show(hbar(top, "total_wpa", "label", f"Most valuable players {season}: total WPA (wins added)"))
    with right:
        clutch = season_impact.nlargest(12, "clutch_wpa").assign(label=lambda d: d.player_name + " · " + d.team)
        show(hbar(clutch, "clutch_wpa", "label", f"Clutch players {season}: WPA on high-leverage balls"))
    st.caption("WPA credits the batter with every ball's change in win probability and debits the bowler. "
               "High-leverage balls are game situations where win probability typically moves 2× more than average.")

    validation = read_json(str(CFG.models / "wpa_validation.json"))
    if validation and not validation.get("skipped"):
        st.subheader("Does WPA measure real skill? Validation against the simulator's hidden ground truth")
        bat, bowl = validation["batting"], validation["bowling"]
        series = [  # (legend name, {category: rho}); a missing category is simply not drawn
            ("WPA per ball", {"Batting skill": bat["wpa_per_ball_vs_true_skill"],
                              "Bowling skill": bowl["wpa_per_ball_vs_true_skill"]}),
            ("Strike rate / economy", {"Batting skill": bat["strike_rate_vs_true_skill"],
                                       "Bowling skill": bowl["economy_vs_true_skill"]}),
            ("Wickets per ball", {"Bowling skill": bowl["wickets_per_ball_vs_true_skill"]}),
        ]
        fig = go.Figure([
            go.Bar(name=name, x=list(vals), y=list(vals.values()), marker_color=T["series"][i], width=0.22,
                   text=[f"{v:.2f}" for v in vals.values()], textposition="outside",
                   textfont=dict(color=T["secondary"]), hovertemplate=name + ": ρ = %{y:.3f}<extra></extra>")
            for i, (name, vals) in enumerate(series)
        ])
        fig = base_layout(fig, "Spearman rank correlation with true skill (1.0 = perfect ranking)", height=340)
        fig.update_layout(bargap=0.3, bargroupgap=0.08)
        fig.update_yaxes(range=[min(0, *[v for _, vals in series for v in vals.values()]) - 0.05, 1.05])
        v1, v2 = st.columns([3, 2])
        with v1:
            show(fig)
        with v2:
            st.metric("Clutch score vs true clutch (batters)",
                      f"ρ = {validation['clutch_batting']['clutch_score_vs_true_clutch']:.2f}",
                      help=f"{validation['clutch_batting']['players']} batters with enough high-leverage balls")
            st.metric("Clutch score vs true clutch (bowlers)",
                      f"ρ = {validation['clutch_bowling']['clutch_score_vs_true_clutch']:.2f}",
                      help=f"{validation['clutch_bowling']['players']} bowlers with enough high-leverage balls")
    with st.expander("Table view: player impact"):
        show_table(season_impact.sort_values("total_wpa", ascending=False)[
            ["player_name", "team", "matches", "total_wpa", "batting_wpa", "bowling_wpa", "clutch_wpa", "runs",
             "strike_rate", "wickets", "economy"]].round(3))

with tabs[2]:
    @st.fragment(run_every="3s")
    def live_panel():
        if not _exists("gold", "live_win_probability"):
            st.info("No live match yet. In two terminals run `make live-score` and `make live-feed`, "
                    "and this panel updates every 3 seconds.")
            return
        live = DeltaTable(CFG.table("gold", "live_win_probability")).to_pandas()
        current = live.sort_values("scored_at").match_id.iloc[-1]
        m = (live[live.match_id == current].sort_values(["innings_no", "delivery_no"])
             .drop_duplicates(["innings_no", "delivery_no"], keep="last").reset_index(drop=True))
        last = m.iloc[-1]
        a, b, c, d = st.columns(4)
        a.metric(f"{last.batting_team}", f"{last.cum_runs}/{last.cum_wickets}", f"{overs(last.legal_balls)} overs",
                 delta_color="off")
        if last.innings_no == 2 and last.target > 0:
            b.metric("Chase", f"need {max(int(last.target - last.cum_runs), 0)} off {int(last.balls_remaining)}")
        else:
            b.metric("Run rate", f"{last.run_rate:.2f}")
        c.metric(f"{last.batting_team} win probability", f"{100 * last.win_prob:.1f}%", f"{100 * last.delta_wp:+.1f} pp")
        d.metric("Momentum swings", int(m.is_swing.sum()))
        show(worm(m, f"LIVE · {current} · {last.venue}"))
        swings = m[m.is_swing].sort_values(["innings_no", "delivery_no"], ascending=False)
        if len(swings):
            st.markdown("**Momentum swing alerts**")
            show_table(pd.DataFrame({
                "Innings": swings.innings_no, "Over": swings.legal_balls.map(overs), "Batter": swings.batter,
                "Bowler": swings.bowler, "Ball": ["WICKET" if w else f"{r} run(s)" for w, r in
                                                  zip(swings.is_wicket, swings.runs_total, strict=True)],
                "ΔWP": (100 * swings.delta_wp).round(1).astype(str) + " pp"}))

    live_panel()

with tabs[3]:
    standings = table("gold", "team_standings")
    tbl = standings[(standings.event_name == league) & (standings.season_year == season)].sort_values("position")
    st.subheader(f"{league} {season}: points table")
    show_table(tbl.assign(team=lambda d: d.team + d.is_champion.map({True: "  🏆 champion", False: ""}))[
        ["position", "team", "played", "won", "lost", "no_result", "points", "net_run_rate"]])
    scoring = table("gold", "season_scoring")
    trend = scoring[scoring.event_name == league].sort_values("season_year")
    s1, s2 = st.columns(2)
    with s1:
        fig = go.Figure(go.Scatter(x=trend.season_year, y=trend.avg_first_innings, mode="lines+markers",
                                   line=dict(width=2, color=T["series"][0]),
                                   marker=dict(size=8, line=dict(width=2, color=T["surface"])),
                                   hovertemplate="%{x}: %{y:.1f} runs<extra></extra>"))
        show(base_layout(fig, "Average first-innings score by season", y_title="Runs"))
    with s2:
        fig = go.Figure(go.Scatter(x=trend.season_year, y=trend.chase_win_pct, mode="lines+markers",
                                   line=dict(width=2, color=T["series"][0]),
                                   marker=dict(size=8, line=dict(width=2, color=T["surface"])),
                                   hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig = base_layout(fig, "Matches won by the chasing side (%)", y_title="%")
        fig.add_hline(y=50, line_width=1, line_color=T["axis"])
        show(fig)
    venues = table("gold", "venue_stats")
    gender = excitement.loc[excitement.event_name == league, "gender"].iloc[0]
    st.subheader("Venue intelligence")
    show_table(venues[venues.gender == gender].sort_values("avg_first_innings", ascending=False)
               .drop(columns=["gender"]))

with tabs[4]:
    if not card:
        st.info("No model card yet. Run the `train` task.")
    else:
        champ, test = card["champion"], card["metrics"][card["champion"]]["test"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Champion model", {"gbt": "GBT", "logistic_regression": "LogReg"}.get(champ, champ),
                  help="Gradient-boosted trees vs logistic regression, chosen on validation log loss")
        m2.metric("Log loss (test)", f"{test['log_loss']:.3f}")
        m3.metric("Brier score (test)", f"{test['brier']:.3f}")
        m4.metric("Calibration error (ECE)", f"{100 * test['ece']:.2f} pp")
        st.caption(f"Train {card['splits']['train']['from']} → {card['splits']['train']['to']} "
                   f"({card['splits']['train']['rows']:,} balls) · validate {card['splits']['valid']['from']} → "
                   f"{card['splits']['valid']['to']} · test {card['splits']['test']['from']} → "
                   f"{card['splits']['test']['to']} ({card['splits']['test']['rows']:,} balls, never seen in training)")
        comparison = pd.DataFrame([
            {"model": name, "split": split, **{k: round(v, 4) for k, v in m[split].items()
                                             if k in ("log_loss", "brier", "brier_skill_score", "auc", "ece",
                                                      "accuracy")}, "train_seconds": m["train_seconds"]}
            for name, m in card["metrics"].items() for split in ("valid", "test")])
        show_table(comparison)
        g1, g2 = st.columns(2)
        with g1:
            calib = pd.DataFrame(test["calibration"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(width=1, color=T["axis"]),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=100 * calib.mean_predicted, y=100 * calib.observed_win_rate,
                                     mode="lines+markers", line=dict(width=2, color=T["series"][0]),
                                     marker=dict(size=8, line=dict(width=2, color=T["surface"])), showlegend=False,
                                     customdata=calib.n,
                                     hovertemplate="predicted %{x:.1f}% → won %{y:.1f}%<br>%{customdata:,} balls"
                                                   "<extra></extra>"))
            fig = base_layout(fig, "Calibration on unseen seasons: predicted vs actual", height=380,
                              x_title="Predicted win probability (%)", y_title="Observed win rate (%)")
            show(fig)
        with g2:
            imp = pd.DataFrame(card["feature_importance"]).head(10)
            show(hbar(imp, "importance", "feature", "What the model looks at (feature importance)", fmt="{:.2f}",
                      height=380))
        prog = pd.DataFrame(card["test_by_match_progress"])
        fig = go.Figure()
        for i, inn in enumerate(sorted(prog.innings_no.unique())):
            p = prog[prog.innings_no == inn]
            fig.add_trace(go.Scatter(x=p.overs, y=100 * p.accuracy, mode="lines+markers", name=f"Innings {inn}",
                                     line=dict(width=2, color=T["series"][i]),
                                     marker=dict(size=8, line=dict(width=2, color=T["surface"])),
                                     hovertemplate=f"Innings {inn}, overs " + "%{x}: %{y:.1f}%<extra></extra>"))
        fig = base_layout(fig, "Accuracy as the match unfolds (test seasons)", x_title="Overs completed",
                          y_title="Correct winner called (%)")
        show(fig)

with tabs[5]:
    dq = table("quality", "dq_results")
    if not dq.empty:
        latest = dq.sort_values("checked_at").groupby("suite").run_id.last()
        view = dq[dq.run_id.isin(latest)].copy()
        state = view.apply(lambda r: "pass" if r.passed else ("fail" if r.severity == "error" else "warn"), axis=1)
        view["status"] = [f"{STATUS[s][0]} {STATUS[s][1]}" for s in state]
        view["failure_rate"] = (100 * view.failure_rate).round(4).astype(str) + "%"
        st.subheader("Latest data-quality results")
        show_table(view[["suite", "status", "check_name", "table_name", "severity", "failing_rows", "total_rows",
                         "failure_rate"]])
    if _exists("silver", "quarantine"):
        q = table("silver", "quarantine")
        st.subheader("Quarantined records")
        show_table(q.groupby("reason").size().rename("records").reset_index())
    history = runs_history()
    if history:
        st.subheader("Pipeline runs")
        show_table(pd.DataFrame([{"run": r["run_id"], "mode": r["mode"], "status": r["status"],
                                  "seconds": r.get("total_seconds"), "tasks": len(r["tasks"])} for r in history]))
        done = [t for t in history[0]["tasks"] if t.get("seconds") is not None]
        if done:
            durations = pd.DataFrame(done)[["name", "seconds"]]
            show(hbar(durations, "seconds", "name", f"Task durations, run {history[0]['run_id']} (seconds)",
                      fmt="{:.1f}s"))
