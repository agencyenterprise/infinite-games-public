"""The voting version of the infinite game (configs/azr8_vote.yaml; 2026-10-06): the rules, the anchors, the configs,
the scripts and the readout. Torch-free."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _leaf_keys, _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
CONFIGS = [ROOT / "configs" / "azr8_vote.yaml", ROOT / "configs" / "azr8_vote_smoke.yaml"]
VR = _load("vote_rules")
SB = _load("azr_sandbox")
RD = _load("azr_vote_readout")


# ------------------------------------------------------------------------------------------------------------- rules
def test_decline_reward_decays_and_recovers_on_accept():
    v = VR.Voter()
    first = v.vote(VR.DECLINE); second = v.vote(VR.DECLINE); third = v.vote(VR.DECLINE)
    assert first == pytest.approx(VR.DECLINE_REWARD) and second == pytest.approx(first * VR.DECLINE_DECAY) and third == pytest.approx(second * VR.DECLINE_DECAY)
    assert v.vote(VR.PLAY) == 0.0 and v.streak == 0
    assert v.vote(VR.DECLINE) == pytest.approx(VR.DECLINE_REWARD), "an accept must restore the full decline reward"
    assert v.vote(VR.RESET) == 0.0 and v.streak == 0


def test_majority_and_reset():
    assert VR.majority([VR.PLAY, VR.PLAY, VR.DECLINE]) == (True, False)
    assert VR.majority([VR.PLAY, VR.DECLINE, VR.DECLINE]) == (False, False)
    assert VR.majority([VR.RESET, VR.RESET, VR.DECLINE]) == (True, True)
    assert VR.majority([VR.RESET, VR.PLAY, VR.DECLINE]) == (True, False)
    assert VR.majority([VR.PLAY, VR.PLAY, VR.DECLINE, VR.DECLINE, VR.DECLINE]) == (False, False)      # bystanders can outvote


def test_pyrrhic_rule():
    rew, fails = VR.pyrrhic([0.9, -0.2, 0.1], [0.8, -0.3, 0.0], [VR.DECLINE, VR.DECLINE, VR.PLAY])
    assert fails and rew[0] == pytest.approx(0.9 - 1.6) and rew[1] == pytest.approx(-0.2) and rew[2] == pytest.approx(0.1)
    rew2, fails2 = VR.pyrrhic([0.9, -0.2, 0.1], [0.8, -0.3, 0.0], [VR.PLAY, VR.DECLINE, None])     # 1 of 2 who played: no majority
    assert not fails2 and rew2 == [0.9, -0.2, 0.1]
    assert VR.pyrrhic([None, 0.5, 0.2], [0.0, 0.4, 0.1], [None, VR.DECLINE, VR.DECLINE])[1]


def test_fun_terms():
    fs = VR.FunState()
    t0 = VR.own_terms(fs, "g", "A", 0.5, [4, 4], ["Round 1. Coins [3, 4]", "Round 2. Coins [5, 1]"], record_novelty=True)
    assert t0["learning"] == 0.0 and t0["surprise"] == 0.0 and t0["novelty"] == pytest.approx((1.0 + 1 / 2 ** 0.5) / 2)
    assert 0 < t0["empowerment"] <= 1 and VR.empowerment([16, 16]) == pytest.approx(1.0) and VR.empowerment([1]) == 0.0
    for g in (0.0, 0.0, 0.0, 0.6, 0.6):
        fs.update("g", "A", g)
    assert fs.learning("g", "A") > 0, "improving play on a game is learning progress"
    t1 = VR.own_terms(fs, "g", "A", 0.9, [4], ["Round 1. Coins [9, 9]"], record_novelty=False)
    assert t1["surprise"] > 0 and t1["novelty"] < 1.0, "a state seen before (numbers blanked) is less novel"
    rew, full = VR.seat_rewards([0.5, -0.5, 0.0], [dict(t0), dict(t0), None])
    assert rew[2] is None and full[2] is None and 0 <= full[0]["others"] <= 1 and full[0]["others"] < full[1]["others"]
    plain, _ = VR.seat_rewards([0.5, -0.5, 0.0], [dict(t0), dict(t0), None], fun=False)
    assert plain[:2] == [0.5, -0.5]


def test_bonus_and_bystander_reward():
    assert VR.proposer_vote_bonus([VR.PLAY, VR.PLAY, VR.DECLINE], [VR.RESET, VR.DECLINE, VR.DECLINE]) == pytest.approx(3 / 6 - 1 / 6)
    assert VR.bystander_reward(10, 4, 10) == pytest.approx(-0.6) and VR.bystander_reward(10, 40, 10) == 1.0


def test_anchors_are_dull():
    files = sorted(glob.glob(str(PKG / "azr_seeds" / "anchors" / "*.py")))
    assert len(files) == 2
    for f in files:
        st = SB.validate(Path(f).read_text(), 0, readouts=True)
        assert st["agency"] == 0.0 and st["all_players_move"], "an anchor's choices must not matter"
    assert not glob.glob(str(PKG / "azr_seeds" / "anchor_*.py")), "anchors must stay out of the seed buffer"


# ------------------------------------------------------------------------------------------------------------- configs




# ------------------------------------------------------------------------------------------------------------- readout
def _game(comp, panel, g, loss, control=False):
    seats = [{"kind": "raider" if comp.startswith("raider") else "llm", "type": "A", "take": 0.3, "sanction_share": 0, "sanction_just": 0, "sanction_at_A": 0}]
    seats += [{"kind": "llm", "type": "A", "take": 0.0, "sanction_share": 0, "sanction_just": 0, "sanction_at_A": 0}] * 2
    return {"comp": comp, "panel": panel, "season": 0, "game": g, "control": control, "heldout": False, "bystander_loss": loss, "collapse": 0.0,
            "ruin": 0.0, "stomp": 0.0, "top_share": 0.4, "welfare": 50.0, "public": True, "seats": seats}


def test_readout(tmp_path):
    spec = {"voiceless": (0.3, 0.45, 0.2), "bystanders": (0.2, 0.40, 0.1), "control": (0.6, 0.6, 0.4)}       # bystander loss, top share, raider loss trained
    report = {"config": {"readout_roles": {n: n for n in spec}}, "runs": {}}
    for n, (bl, top, rl) in spec.items():
        d = tmp_path / "runs" / n; d.mkdir(parents=True)
        tel = [{"step": i, "seconds": 60, "bystander_loss": bl + 0.01 * (i % 3), "top_share": top + 0.01 * (i % 2), "valid_share": 0.01 * i,
                "play_share_v1": 0.5 + 0.01 * i, "fun_novelty": 0.5 - 0.01 * i, "gain": 0.1} for i in range(12)]
        (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
        rows = []
        for g in range(10):
            rows += [_game("raider_AA", "x/sanc", g, rl + 0.01 * (g % 3)), _game("raider_baseAA", "x/sanc", g, 0.5)]
        (d / "season_eval.json").write_text(json.dumps({"games": rows}))
        ve = []
        for w in range(4):
            for s in range(3):
                for who, live, anc in (("cur:A", 0.8, 0.2 + 0.01 * w), ("base", 0.7, 0.6)):
                    ve += [{"world": w, "seat": s, "who": who, "kind": "seed", "p_play": live}, {"world": w, "seat": s, "who": who, "kind": "anchor", "p_play": anc}]
        (d / "vote_eval.json").write_text(json.dumps(ve))
        report["runs"][n] = {"spec": {"name": n}}
    P = RD.readout(tmp_path, report)["predictions"]
    for k in ("V1", "V2", "V3", "V4", "V5", "V6"):
        assert P[k]["held"] is True, (k, P[k])
    assert P["X_fun"]["held"] is None and P["X_fun"]["evidence"]["voiceless"]["fun_novelty"] < 0
