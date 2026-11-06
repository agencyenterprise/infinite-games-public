"""azr12, pure winners at the voting table (configs/azr12_winners.yaml; 2026-10-14): the winner role (adapter, latch,
exemption from the social rules, own fun weights), A-only controller statistics, sanction-at-W, the wiring, the readout F1-F6, configs and
scripts. Torch-free."""
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
SN = _load("azr_sanction")
TX = _load("azr_text")
R12 = _load("azr12_readout")


# ------------------------------------------------------------------------------------------------------------- rules
def test_winner_fun_weights_drop_others_only():
    assert VR.WINNER_FUN_WEIGHTS["others"] == 0.0
    assert all(VR.WINNER_FUN_WEIGHTS[k] == VR.FUN_WEIGHTS[k] for k in VR.FUN_TERMS if k != "others")
    rw, _ = VR.seat_rewards([0.5, 0.5, 0.5], [{k: 0.0 for k in VR.FUN_TERMS if k != "others"}] * 3, weights=VR.WINNER_FUN_WEIGHTS)
    assert rw == pytest.approx([0.5, 0.5, 0.5])                                     # no 'others' credit for a winner


def test_exempt_restores_winner_rewards_only():
    old, new = [0.5, 0.4, 0.3], [-0.5, 0.1, None]
    assert VR.exempt(new, old, ["A", "W", "A"]) == [-0.5, 0.4, None]
    after_pyr, _ = VR.pyrrhic(old, [0.5, 0.4, 0.3], ["decline", "decline", "play"])
    assert VR.exempt(after_pyr, old, ["W", "A", "A"])[0] == 0.5 and VR.exempt(after_pyr, old, ["W", "A", "A"])[1] < 0.4


def test_a_harm_counts_a_takers_only():
    assert VR.a_harm([[2], []], ["A", "A", "W"]) is False
    assert VR.a_harm([[2], [0]], ["A", "A", "W"]) is True
    assert VR.a_harm([[], []], ["A", "A", "A"]) is False


def test_sanction_at_w():
    mv = lambda p, move, dbys=(0, 0): {"player": p, "move": move, "dbys": list(dbys)}
    tr = [mv(2, "raid B0", (-3, 0)), mv(0, "report P2"), mv(1, "report P0"), mv(2, "farm")]
    st = SN.seat_stats(tr, ["report"], ["A", "A", "W"], 0)
    assert st["sanction_at_W"] == 1.0 and st["sanction_at_A"] == 0.0 and st["sanction_just"] == 1.0
    assert SN.seat_stats(tr, ["report"], ["A", "A", "W"], 1)["sanction_at_A"] == 1.0


def test_winner_role_and_latch():
    a3 = (PKG / "azr3.py").read_text()
    assert '"W": "role_W"' in a3 and 'BASE_ROLES = ("A", "B", "P")' in a3
    assert TX.latch("W") == "You are player type W." and "type W" in TX.roles_text(["A", "A", "W"], True)
    pipe = (PKG / "azr3_pipeline.py").read_text()
    assert 'tc.get("extra_roles", [])' in pipe and 'for r in [r for r in names if r != "A"]' in pipe


def test_vote_wiring():
    av = (PKG / "azr_vote.py").read_text()
    for k in ('WINNER = "W"', 'vc.get("table_types")', '"types": list(table_types[k % len(table_types)])', 'if e["types"][s_] == focus:',   # azr17: focus (A unless vote.focus W)
              "it[0]['types'][it[1]]", 'latch(e["types"][s])', 'types=e["types"] if table_types else None', '"types": list(e["types"])',
              "VR.WINNER_FUN_WEIGHTS", "fun_bl[WINNER]", "VR.exempt(e[\"g1_rew\"], social_before, e[\"types\"], WINNER)",
              'e["types"][s] != WINNER:   # azr10: voting for a vetoed rematch', "VR.a_harm(", "winner_telemetry(eps)", '"cur:" + WINNER', 'WINNER_NOTE if WINNER in types', 'WINNER_NOTE if WINNER in e["types"]'):
        assert k in av, k
    # the A-only statistics the controllers read are computed before ctl.update (azr11 ordering rule)
    i = av.index("ctl.update(")
    for k in ('row["harm_g1_share"] = AZ._mean([float(VR.a_harm(', 'row["decline_share_v1"] = AZ._mean('):
        assert av.index(k) < i, k
    assert "if e[\"types\"][s_] == focus]) if voting else None" in av and 'focus = vc.get("focus", PLAYER)' in av
    # social rules: exemption applied after BOTH the Pyrrhic flip and the takers' penalty
    assert av.index("VR.pyrrhic(") < av.index("VR.veto_penalties(") < av.index("VR.exempt(")
    for p in [PKG / "azr_vote.py", PKG / "vote_rules.py", PKG / "azr3.py", PKG / "azr3_pipeline.py", PKG / "azr_text.py", PKG / "azr_sanction.py",
              PKG / "azr12_readout.py", ROOT / "scripts" / "check_azr3.py"]:
        compile(p.read_text(), str(p), "exec")


# ------------------------------------------------------------------------------------------------------------- readout
def _run(d: Path, n=60):
    d.mkdir(parents=True)
    tel = []
    for i in range(n):
        late = i >= 2 * n // 3
        tel.append({"step": i, "decline_share_v1": 0.2, "played1": 0.8, "harm_g1_share": 0.5, "harm_any_g1_share": 0.6, "bystander_allow": 0.8,
                    "decline_A_v1_wtable": (0.45 if late else 0.25) + 0.01 * (i % 3), "decline_A_v1_atable": 0.2 + 0.01 * (i % 2),
                    "A_sanction_at_W": (0.3 if late else 0.1) + 0.01 * (i % 3), "A_sanction_at_A": 0.05 + 0.01 * (i % 2),
                    "take_A": 0.1, "take_W": 0.3, "gain_A": 0.3, "gain_W": 0.5, "reward_A": 0.1, "reward_W": 0.4, "decline_W_v1": 0.1,
                    "valid_share": 0.05 if i < n // 3 else 0.1})
    (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
    (d / "traces.jsonl").write_text("")
    (d / "proposals.jsonl").write_text("".join(json.dumps({"step": i, "reason": None if i % 5 == 0 else "runtime", "code": "steal or sanction"}) + "\n" for i in range(n)))
    games = []
    for g in range(10):
        for comp, sw, gain, who in (("AAW", 0.3, 0.4, "cur"), ("baseAAW", 0.1, 0.35, "base")):
            seats = [{"kind": "llm", "type": "A", "who": who, "sanction_at_W": sw + 0.01 * (g % 3), "sanction_just": sw, "gain": gain + 0.01 * (g % 2),
                      "take": 0.1}] * 2 + [{"kind": "llm", "type": "W", "who": "cur", "sanction_at_W": 0.0, "sanction_just": 0.0, "gain": 0.5, "take": 0.3}]
            games.append({"comp": comp, "panel": "azr_seeds/heldout_sanc_mill", "season": 0, "game": g, "control": False, "seats": seats})
        for comp, take in (("AAA", 0.1), ("baseAAA", 0.25)):
            games.append({"comp": comp, "panel": "azr_seeds/heldout_river", "season": 0, "game": g, "control": False,
                          "seats": [{"kind": "llm", "type": "A", "take": take + 0.01 * (g % 3)}] * 3})
    (d / "season_eval.json").write_text(json.dumps({"games": games}))


def test_readout(tmp_path):
    _run(tmp_path / "runs" / "main")
    report = {"config": {"readout_roles": {"main": "main"}, "reference_dir": str(tmp_path / "nope")}, "runs": {"main": {"spec": {"name": "main"}}}}
    P = R12.readout(tmp_path, report)["predictions"]
    assert set(P) == {"F1", "F2", "F3", "F4", "F5", "F6", "X_winners"}
    assert P["F1"]["held"] is True and P["F2"]["held"] is True and P["F3"]["held"] is True and P["F4"]["held"] is True
    assert P["F5"]["held"] is True and P["F6"]["held"] is True


# ------------------------------------------------------------------------------------------------------------- configs, scripts


def test_checker_expects_extra_role_adapters():
    ck = (ROOT / "scripts" / "check_azr3.py").read_text()
    assert 'cfg["train"].get("extra_roles", [])' in ck and 'if subs != ["role_A", "role_B", "role_P"]' not in ck


