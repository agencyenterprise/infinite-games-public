"""azr20 (Thomas 2026-11-01): azr19 + bystanders as a bare majority of voters (two voting seats per bystander) + bystander reward as a share of
its own holdings. Torch-free."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _load  # noqa: E402

VR = _load("vote_rules")
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
PLAYERS, BYSTANDERS = 3, 2


def _cfg(name):
    return yaml.safe_load(open(CFG / f"{name}.yaml"))


def test_share_reward_counts_losing_everything_as_minus_one():
    assert VR.bystander_share_reward(2, 0) == -1.0
    assert VR.bystander_share_reward(10, 0) == -1.0
    assert VR.bystander_share_reward(10, 5) == -0.5
    assert VR.bystander_share_reward(0, 0) == 0.0
    assert VR.bystander_share_reward(4, 6) == 0.5
    assert abs(VR.plant_share_reward(2, 0, 0.2) - (-0.8)) < 1e-12


def test_old_reward_unchanged():
    assert VR.bystander_reward(2, 0, 30.0) == -2 / 30
    assert VR.plant_reward(2, 0, 30.0, 0.2) == 0.2 - 2 / 30


def test_two_seats_make_the_bystanders_a_bare_majority():
    y, n = VR.PLAY, VR.DECLINE
    seats = 2
    assert VR.majority([y, y, y] + [n] * (BYSTANDERS * seats))[0] is False      # both bystanders refuse: blocked
    assert VR.majority([y, y, y] + [n, n, y, y])[0] is True                      # one bystander refuses: players decide
    assert VR.majority([n, n, n] + [y] * (BYSTANDERS * seats))[0] is True       # both bystanders allow: played
    assert VR.majority([y, y, y] + [n, n])[0] is True                            # azr19 (one seat each): bystanders cannot block


def test_azr20_is_azr19_plus_seats_and_share():
    for v in ("inf", "fin", "mix", "smoke"):
        a, b = _cfg(f"azr19_{v}"), _cfg(f"azr20_{v}")
        assert b["run_id"] == a["run_id"].replace("azr19", "azr20")
        assert b["output_base"] == a["output_base"].replace("azr19", "azr20")
        for r in b["runs"]:
            assert r["vote"].pop("bystander_seats") == 2 and r["vote"].pop("bystander_share") is True
            assert r["vote"]["bystanders_vote"] is True and r["vote"]["harm_veto"] is False
        for c in (a, b):
            c.pop("run_id"); c.pop("output_base")
        assert a == b, v
