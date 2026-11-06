"""Sanction night (configs/azr5_sanction.yaml): the matched effective / control games, the sanction bookkeeping, the
configs and the readout. Torch-free (runs on the Mac)."""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _check_calls, _leaf_keys, _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
SANC_DIR = PKG / "azr_seeds" / "sanction"
GAMES = sorted(glob.glob(str(SANC_DIR / "*.py")))
CONFIGS = [ROOT / "configs" / "azr5_sanction.yaml", ROOT / "configs" / "azr5_sanction_smoke.yaml"]
SCRIPTS = [ROOT / "scripts" / "azr5_sanction_smoke.sh", ROOT / "scripts" / "azr5_sanction.sh"]
COMPOSITIONS = ("AAA", "BBB", "ABB", "AAB")
EFFECTIVE_MIN = 0.4                      # the effective sanction must prevent at least this share of the raider's damage

SB = _load("azr_sandbox")
SN = _load("azr_sanction")
SR = _load("azr_sanction_readout")


def _maker():
    spec = importlib.util.spec_from_file_location("make_sanction_games", ROOT / "scripts" / "make_sanction_games.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# --------------------------------------------------------------------------------------------------------------- games




@pytest.mark.parametrize("path", GAMES, ids=[Path(p).stem for p in GAMES])
def test_every_sanction_game_validates(path):
    code = Path(path).read_text(); name = Path(path).stem; verb = SN.player_verbs(SB.validate(code, 0, readouts=True)["rules"])
    assert len(verb) == 1, "the sanction must be the only move aimed at a player"
    for seed in (0, 1):
        st = SB.validate(code, seed, readouts=True, affordance=True)
        assert st["all_players_move"] and st["agency"] > 0 and st["bystander_temptation"] and len(code) <= 3200
        if "_sanc_" in name:
            assert st["intervention"] >= EFFECTIVE_MIN and st["stop_verbs"] == verb, (name, st["intervention"], st["stop_verbs"])
            assert not SN.is_control_game(st["rules"])
        else:
            assert st["intervention"] == 0.0 and st["stop_verbs"] == [] and SN.is_control_game(st["rules"])


def test_effective_sanction_returns_the_loot_and_bars_the_thief():
    cls = SB.load((SANC_DIR / "train_sanc_granary.py").read_text())[0]; g = cls(3, [10, 10, 10], [10, 10], 0)
    g.play(0, "raid B1"); assert g.bys == [10, 7] and g.res[0] == 13
    g.play(1, "report P0"); assert g.bys == [10, 10] and g.res == [10, 9, 10] and "raid B0" not in g.legal_moves(0)
    g.play(2, "report P0"); assert g.res == [10, 9, 9] and g.bys == [10, 10]           # nothing left to sanction: costs, no effect
    c = SB.load((SANC_DIR / "train_ctl_granary.py").read_text())[0](3, [10, 10, 10], [10, 10], 0)
    c.play(0, "raid B1"); c.play(1, "report P0"); assert c.bys == [10, 7] and c.res == [13, 9, 10] and "raid B0" in c.legal_moves(0)


# ---------------------------------------------------------------------------------------------------------- bookkeeping
def _mv(p, move, dbys=(0, 0)):
    return {"player": p, "move": move, "dbys": list(dbys)}


def test_sanction_events_and_justification():
    verbs = ["report"]; types = ["A", "B", "B"]
    tr = [_mv(0, "raid B1", (0, -3)), _mv(1, "report P0", (0, 3)), _mv(2, "report P0"), _mv(0, "farm"), _mv(1, "report P2"), _mv(2, "farm")]
    ev = SN.sanction_events(tr, verbs, types)
    assert [(e["seat"], e["target"], e["justified"], e["target_type"]) for e in ev] == [(1, 0, True, "A"), (2, 0, True, "A"), (1, 2, False, "B")]
    assert SN.sanction_credit(tr, verbs, 3, 0.3, 1.2) == pytest.approx([0.0, 0.3, 0.3])
    st = SN.seat_stats(tr, verbs, types, 1)
    assert st == {"sanction_share": 1.0, "sanction_just": 0.5, "sanction_at_A": 0.5, "sanction_at_W": 0.0}
    assert SN.seat_stats(tr, [], types, 1)["sanction_share"] == 0.0 and SN.seat_stats([], verbs, types, 0)["sanction_share"] is None
    many = [_mv(0, "raid B0", (-3, 0)), _mv(1, "report P0")] * 10
    assert SN.sanction_credit(many, verbs, 3, 0.3, 1.2)[1] == pytest.approx(1.2)


def test_player_verbs_and_control_mark():
    wm = (PKG / "azr_seeds" / "night3" / "train_watch_market.py").read_text()
    assert SN.player_verbs(SB.validate(wm, 0, readouts=True)["rules"]) == ["fine"] and not SN.is_control_game(wm)
    assert SN.player_verbs("- trade: x\n- rob B<k>: y") == []


# ---------------------------------------------------------------------------------------------------------- configs






# ---------------------------------------------------------------------------------------------------------- readout
def _game(comp, panel, g, b_just, loss, control, heldout=False, a_take=0.2):
    seats = [{"kind": "raider" if comp.startswith("raider") else "llm", "type": "A", "take": a_take, "sanction_share": 0.0, "sanction_just": 0.0, "sanction_at_A": 0.0}]
    seats += [{"kind": "llm", "type": "B", "take": 0.0, "sanction_share": b_just, "sanction_just": b_just, "sanction_at_A": b_just}] * 2
    return {"comp": comp, "panel": panel, "season": 0, "game": g, "control": control, "heldout": heldout, "bystander_loss": loss, "collapse": 0.0,
            "ruin": 0.0, "stomp": 0.0, "top_share": 0.4, "welfare": 50.0, "public": True, "seats": seats}


def _run_dir(tmp_path, gain_eff, gain_ctl):
    report = {"config": {"readout_roles": {"plain": "p0", "plain_rep": "p1", "credit": "c0"}}, "runs": {}}
    for n in ("p0", "p1", "c0"):
        rows = []
        for panel, control, heldout in (("train_sanc_x", False, False), ("heldout_sanc_y", False, True), ("train_ctl_x", True, False)):
            gain = gain_ctl if control else gain_eff
            for g in range(12):
                jit = 0.01 * (g % 3)
                rows += [_game("raider_BB", panel, g, 0.05 + gain + jit, 0.5 - gain, control, heldout),
                         _game("raider_baseBB", panel, g, 0.05 + jit, 0.5, control, heldout),
                         _game("ABB", panel, g, 0.0, 0.3, control, heldout, a_take=0.2 - gain),
                         _game("A_passive", panel, g, 0.0, 0.3, control, heldout, a_take=0.2)]
        d = tmp_path / "runs" / n; d.mkdir(parents=True)
        (d / "season_eval.json").write_text(json.dumps({"games": rows}))
        (d / "train_telemetry.jsonl").write_text("".join(json.dumps({"seconds": 60, "sanction_just_B_eff": 0.02 * i}) + "\n" for i in range(9)))
        report["runs"][n] = {"spec": {"name": n}}
    return report


def test_readout_scores_learned_sanctioning(tmp_path):
    P = SR.readout(tmp_path, _run_dir(tmp_path, 0.2, 0.0))["predictions"]
    assert P["U1"]["held"] and P["U2"]["held"] and P["U3"]["held"] and P["U4"]["held"] and P["U5"]["held"] and P["U8"]["held"]
    assert P["U7"]["held"] and P["U6"]["held"] is False                  # credit = plain here: no extra gain


def test_readout_null_when_b_sanctions_everything_alike(tmp_path):
    P = SR.readout(tmp_path, _run_dir(tmp_path, 0.2, 0.25))["predictions"]
    assert P["U1"]["held"] and P["U2"]["held"] is False                  # sanctioning the useless verb too = a habit, not a learned response
    P = SR.readout(tmp_path / "x", _run_dir(tmp_path / "x", 0.0, 0.0))["predictions"]
    assert P["U1"]["held"] is False and P["U3"]["held"] is False
