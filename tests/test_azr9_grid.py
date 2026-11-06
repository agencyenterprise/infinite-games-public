"""The azr9 grid (configs/azr9_*.yaml; 2026-10-07): the new vote-rule knobs (decline size, centred fun, plant
bystanders), the liveliness criteria and selection, the deep-config writer, the deep readout, configs and scripts. Torch-free."""
from __future__ import annotations

import importlib.util
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
G = _load("azr9_grid")
DR = _load("azr9_deep_readout")


def _selector():
    spec = importlib.util.spec_from_file_location("select_azr9", ROOT / "scripts" / "select_azr9.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ------------------------------------------------------------------------------------------------------------- rules
def test_defaults_are_the_night8_rules():
    assert VR.decline_reward(0) == pytest.approx(0.1) and VR.decline_reward(2) == pytest.approx(0.025)
    v = VR.Voter(); assert v.vote(VR.DECLINE) == pytest.approx(0.1)
    t = {"learning": 0.5, "novelty": 0.5, "surprise": 0.0, "empowerment": 1.0}
    a, _ = VR.seat_rewards([0.3, 0.1, 0.0], [dict(t), dict(t), None]); b, _ = VR.seat_rewards([0.3, 0.1, 0.0], [dict(t), dict(t), None], baseline=0.0)
    assert a == b


def test_decline_size_and_decay_per_voter():
    v = VR.Voter(base=0.4, decay=0.25)
    assert [v.vote(VR.DECLINE) for _ in range(3)] == pytest.approx([0.4, 0.1, 0.025])
    assert v.vote(VR.PLAY) == 0.0 and v.vote(VR.DECLINE) == pytest.approx(0.4)          # an accept resets the run
    assert VR.Voter(base=0.0).vote(VR.DECLINE) == 0.0                                    # a plant's refusal pays nothing


def test_centred_fun_subtracts_the_baseline_from_the_fun_total_only():
    t = {"learning": 0.2, "novelty": 0.4, "surprise": 0.0, "empowerment": 0.6}
    raw, full = VR.seat_rewards([0.3, -0.2, 0.0], [dict(t), dict(t), None])
    cen, _ = VR.seat_rewards([0.3, -0.2, 0.0], [dict(t), dict(t), None], baseline=VR.fun_total(full[0]))
    assert cen[0] == pytest.approx(0.3) and cen[1] == pytest.approx(-0.2 + VR.fun_total(full[1]) - VR.fun_total(full[0])) and cen[2] is None
    assert raw[0] == pytest.approx(0.3 + VR.fun_total(full[0]))
    nofun, _ = VR.seat_rewards([0.3, -0.2, 0.0], [dict(t), dict(t), None], fun=False, baseline=5.0)
    assert nofun[:2] == [0.3, -0.2]                                                     # the control is untouched by the baseline


def test_fun_baseline_is_an_ema():
    fs = VR.FunState(); assert fs.fun_baseline("A") == 0.0
    fs.update_fun("A", 0.3); assert fs.fun_baseline("A") == pytest.approx(0.3)
    fs.update_fun("A", 0.1); assert fs.fun_baseline("A") == pytest.approx(0.3 * (1 - VR.EMA_FUN) + 0.1 * VR.EMA_FUN)


def test_plant_reward():
    assert VR.plant_reward(10, 10, 10, 0.2) == pytest.approx(0.2)                        # untouched: the growth
    assert VR.plant_reward(10, 6, 10, 0.2) == pytest.approx(-0.2)                        # harmed more than it grows
    assert VR.plant_reward(10, 0, 10, 0.05) == pytest.approx(-0.95)




# ------------------------------------------------------------------------------------------------------------- grid
def _trace(step, game, bys_votes, loss, source="seed"):
    return {"step": step, "game": game, "source": source, "v1": ["play"] * 3 + bys_votes, "v2": None,
            "g1": {"bystander_loss": loss} if loss is not None else None}


def _cell(d: Path, decline, allow, taste_cur, harmful_allow, steps=30):
    d.mkdir(parents=True)
    tel = [{"step": i, "decline_share_v1": decline, "bystander_allow": allow, "reset_share": 0.02, "played1": 0.9, "valid_share": 0.05,
            "fun_baseline": 0.2, "bystander_reward": 0.1} for i in range(steps)]
    (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
    tr = []
    for s in range(steps):
        for k, loss in enumerate((0.0, 0.2, 0.5, 0.9)):
            a = "play" if (loss < 0.4 or harmful_allow) else "decline"
            tr.append(_trace(s, f"game{k}", [a, a], loss))
    (d / "traces.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tr))
    ve = []
    for w in range(2):
        for s in range(3):
            for who, live, anc in (("cur:A", 0.8, 0.8 - taste_cur), ("base", 0.8, 0.79)):
                ve += [{"world": w, "seat": s, "who": who, "kind": "seed", "p_play": live}, {"world": w, "seat": s, "who": who, "kind": "anchor", "p_play": anc}]
    (d / "vote_eval.json").write_text(json.dumps(ve))


def test_cell_metrics_and_selection(tmp_path):
    _cell(tmp_path / "alive", 0.2, 0.5, 0.3, harmful_allow=False)
    _cell(tmp_path / "saturated", 0.0, 0.0, 0.0, harmful_allow=True)
    _cell(tmp_path / "half", 0.2, 0.95, 0.1, harmful_allow=True)
    cells = {n: G.cell_metrics(tmp_path / n) for n in ("alive", "saturated", "half")}
    assert cells["alive"]["met"] == {"decline": True, "allow": True, "taste": True, "harm": True} and cells["alive"]["score"] == 4
    assert cells["alive"]["harm"] < 0 and cells["alive"]["harm_games"] == 4
    assert cells["saturated"]["score"] == 0 and cells["half"]["met"]["allow"] is False
    assert G.select(cells, 2) == ["alive", "half"]


def test_harm_corr_needs_enough_games():
    tr = [_trace(0, "a", ["play", "play"], 0.1), _trace(0, "b", ["decline", "decline"], 0.8)]
    assert G.harm_corr(tr) == (None, 2)




def test_deep_readout(tmp_path):
    for n, dec, harmful in (("deep_a", 0.2, False), ("deep_b", 0.15, False), ("control", None, True)):
        _cell(tmp_path / "runs" / n, dec, 0.5, 0.3, harmful_allow=harmful)
        (tmp_path / "runs" / n / "proposals.jsonl").write_text("".join(json.dumps({"step": s, "reason": None if s % 5 == 0 else "syntax",
                                                                                    "code": 'RULES = """take from B0"""\n' * 3}) + "\n" for s in range(30)))
        (tmp_path / "runs" / n / "season_eval.json").write_text(json.dumps({"games": []}))
    report = {"config": {"readout_roles": {"pick1": "deep_a", "pick2": "deep_b", "control": "control"}},
              "runs": {n: {"spec": {"name": n}} for n in ("deep_a", "deep_b", "control")}}
    ro = DR.readout(tmp_path, report); P = ro["predictions"]
    assert P["W1"]["held"] is True and P["W3"]["held"] is True and set(P) >= {"W1", "W2", "W3", "W4", "W5", "W6", "X_proposer"}
    sh = ro["runs"]["deep_a"]["proposer_shape"]; assert sh["first"]["valid"] == 2 and sh["first"]["mention_bystanders"] == 1.0


# ------------------------------------------------------------------------------------------------------------- configs, scripts




def test_rows_strip_nul_prefix(tmp_path, capsys):
    """2026-10-08: the box's filesystem prefixed one append-only telemetry file with 5 MiB of NULs; the rows after them were intact."""
    p = tmp_path / "t.jsonl"; p.write_bytes(b"\x00" * 1000 + b'{"step": 0}\n{"step": 1}\n\n')
    assert G.rows(p) == [{"step": 0}, {"step": 1}] and "stripped 1000 NUL bytes" in capsys.readouterr().out
    assert G.rows(tmp_path / "missing.jsonl") == []
