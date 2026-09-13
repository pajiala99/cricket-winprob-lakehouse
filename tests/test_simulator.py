import json

from cricklake.ingest import simulator as sim
from tests.conftest import simulated_matches


def test_outcome_probabilities_are_valid_and_respond_to_skill():
    average = sim.outcome_probs("middle", bat=0.0, bowl=0.0, aggression=0.0, conditions=0.0)
    star = sim.outcome_probs("middle", bat=1.5, bowl=0.0, aggression=0.0, conditions=0.0)
    assert abs(sum(average) - 1) < 1e-9 and abs(sum(star) - 1) < 1e-9
    assert star[5] > average[5]                            # better batters hit more sixes
    assert star[sim.WICKET_IDX] < average[sim.WICKET_IDX]  # ... and get out less


def test_matches_are_internally_consistent():
    for match in simulated_matches(n=10, seed=11):
        info, innings = match["info"], match["innings"]
        assert len(info["teams"]) == 2 and info["match_type"] == "T20"
        totals = []
        for inn in innings[:2]:
            legal = 0
            for over in inn["overs"]:
                for d in over["deliveries"]:
                    assert d["runs"]["total"] == d["runs"]["batter"] + d["runs"]["extras"]
                    legal += not ("wides" in d.get("extras", {}) or "noballs" in d.get("extras", {}))
            assert legal <= 121  # 120 + a rare umpire miscount
            totals.append(sum(d["runs"]["total"] for o in inn["overs"] for d in o["deliveries"]))
        target = innings[1]["target"]["runs"]
        outcome = info["outcome"]
        if outcome.get("winner") == innings[1]["team"]:
            assert totals[1] >= target
        elif outcome.get("winner") == innings[0]["team"] and "method" not in outcome:
            assert totals[1] < target - 1 and outcome["by"]["runs"] == target - 1 - totals[1]


def test_simulation_is_deterministic():
    a = json.dumps(simulated_matches(n=2, seed=3))
    b = json.dumps(simulated_matches(n=2, seed=3))
    assert a == b
