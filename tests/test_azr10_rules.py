"""The azr10 grid (configs/azr10_*.yaml; 2026-10-10): valid games must move holdings (azr_sandbox require_effect), the
harmed bystander's rematch veto with a penalty on its takers (vote_rules), the readout's W7-W9, configs and scripts. Torch-free."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
CFG = ROOT / "configs"
VR = _load("vote_rules")
SB = _load("azr_sandbox")
R10 = _load("azr10_deep_readout")

WORKING = (PKG / "azr_seeds" / "anchors" / "anchor_coinflip.py").read_text()
EMPTY = WORKING.replace("""        if (self.turn + 1) % self.n == 0:
            w = self.rng.randrange(self.n)
            for q in range(self.n):
                if q != w and self.res[q] > 0:
                    self.res[q] -= 1; self.res[w] += 1
""", "")


# ------------------------------------------------------------------------------------------------------------- require_effect
def test_require_effect_rejects_empty_games_only():
    assert EMPTY != WORKING
    ok = SB.validate(WORKING, seed=3, require_effect=True); assert ok["effect"] >= SB.EFFECT_MIN
    assert SB.validate(EMPTY, seed=3)["effect"] == 0.0                           # still valid under the old rule
    with pytest.raises(SB.SandboxError) as e:
        SB.validate(EMPTY, seed=3, require_effect=True)
    assert e.value.reason == "no_effect"


def test_require_effect_is_threaded_through_the_pool_job():
    r = SB._validate_job((EMPTY, 3, False, False, True)); assert r["ok"] is False and r["reason"] == "no_effect"
    assert SB._validate_job((EMPTY, 3, False, False))["ok"] is True              # old 4-tuple: old rule


# ------------------------------------------------------------------------------------------------------------- harm veto
def test_harmed_and_takers():
    assert VR.harmed([10, 10], [7, 10]) == [True, False]
    trace = [{"player": 0, "dbys": [-2, 0]}, {"player": 1, "dbys": [0, 0]}, {"player": 2, "dbys": [-1, 0]}, {"player": 0, "dbys": [0, 1]}]
    assert VR.takers(trace, 2) == [[0, 2], []]


def test_harm_veto_needs_harm_and_a_refusal():
    votes = ["play", "play", "play", "decline", "decline"]
    assert VR.harm_veto(votes, [True, False], 3) == [0]                          # B1 refused but was not harmed
    assert VR.harm_veto(["play"] * 3 + ["play", "decline"], [True, False], 3) == []
    assert VR.harm_veto(["play"] * 3, [True, True], 3) == []                     # voiceless bystanders cannot veto


def test_veto_penalties_hit_only_the_takers():
    out = VR.veto_penalties([0.5, 0.2, None], [[0, 2], [1]], [0], 0.3)
    assert out == pytest.approx([0.2, 0.2, None]) or (out[0] == pytest.approx(0.2) and out[1] == pytest.approx(0.2) and out[2] is None)
    out2 = VR.veto_penalties([0.5, 0.2, 0.1], [[0], [0]], [0, 1], 0.3); assert out2[0] == pytest.approx(-0.1)


def test_rematch_voter_penalties():
    assert VR.rematch_voter_penalties(["play", "decline", "reset", "decline", "decline"], [0], 3, 0.3) == [0.3, 0.0, 0.3]
    assert VR.rematch_voter_penalties(["play", "play", "play"], [], 3, 0.3) == [0.0, 0.0, 0.0]          # no veto, no penalty


def test_vote_loop_wires_the_new_rules():
    av = (PKG / "azr_vote.py").read_text()
    for k in ('"harm_veto"', '"veto_penalty"', '"require_effect"', "VR.harm_veto(", "VR.veto_penalties(", "VR.harmed(", "VR.takers(",
              "require_effect=require_effect", "VETO_NOTE", "VR.rematch_voter_penalties(", '"veto_voter_penalty"', "world_text(w, veto)", '"veto_share"', '"harm_g1_share"', '"valid_effect"', '"vetoed"'):
        assert k in av, k
    for p in [PKG / "azr_vote.py", PKG / "vote_rules.py", PKG / "azr_sandbox.py", PKG / "azr10_deep_readout.py", ROOT / "scripts" / "check_azr3.py"]:
        compile(p.read_text(), str(p), "exec")


# ------------------------------------------------------------------------------------------------------------- readout
def test_readout_w7_w9(tmp_path):
    def run(n, take_first, take_last, empty):
        d = tmp_path / "runs" / n; d.mkdir(parents=True)
        tel = [{"step": i, "decline_share_v1": 0.2, "bystander_allow": 0.5, "veto_share": 0.1, "harm_g1_share": 0.3, "valid_share": 0.1} for i in range(30)]
        (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
        tr = []
        for s in range(30):
            tk = take_first if s < 10 else take_last
            mv = [{"player": 0, "dbys": [-1, 0]}] * int(tk * 10) + [{"player": 1, "dbys": [0, 0]}] * (10 - int(tk * 10))
            tr.append({"step": s, "game": f"g{s % 5}", "source": "seed", "v1": ["play"] * 5, "g1": {"bystander_loss": tk, "gains": [0.1, 0, 0], "moves": mv}})
            tr.append({"step": s, "game": f"p{s}", "source": "proposal", "v1": ["play"] * 5,
                       "g1": {"bystander_loss": 0.0, "gains": [0, 0, 0] if s % 10 < empty * 10 else [0.1, 0, 0], "moves": []}})
        (d / "traces.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tr))
        games = []
        for g in range(8):
            for comp, take in (("AAA", 0.1 if n != "control" else 0.4), ("baseAAA", 0.3)):
                games.append({"comp": comp, "panel": "x", "season": 0, "game": g, "control": False, "bystander_loss": take,
                              "seats": [{"kind": "llm", "take": take + 0.01 * (g % 3)}] * 3})
        (d / "season_eval.json").write_text(json.dumps({"games": games}))
        (d / "proposals.jsonl").write_text("")
    run("deep_a", 0.5, 0.1, 0.0); run("deep_b", 0.4, 0.1, 0.1); run("control", 0.3, 0.3, 0.6)
    report = {"config": {"readout_roles": {"pick1": "deep_a", "pick2": "deep_b", "control": "control"}},
              "runs": {n: {"spec": {"name": n}} for n in ("deep_a", "deep_b", "control")}}
    P = R10.readout(tmp_path, report)["predictions"]
    assert P["W7"]["held"] is True and P["W8"]["held"] is True and P["W9"]["held"] is True and "X_veto" in P and "W1" in P


# ------------------------------------------------------------------------------------------------------------- configs, scripts


