"""azr15 (configs/azr15_predict.yaml; 2026-10-19): bands, part-by-part prediction prompts, the mix, memory notes, the fun share,
the fun-share controller, the trainer wiring, the readout J1-J5, configs and scripts. Torch-free."""
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
VP = _load("vote_predict")
VC = _load("vote_control")
VR = _load("vote_rules")
R15 = _load("azr15_readout")


# ------------------------------------------------------------------------------------------------------------- bands
def test_binner_edges_and_bands():
    b = VP.Binner(window=100, min_n=10)
    assert not b.ready("A") and b.edges("A", "gain") is None
    for x in range(100):
        b.add("A", "gain", x / 100); b.add("A", VP.TOTAL, x / 100)
    assert b.ready("A") and not b.ready("W")
    ed = b.edges("A", "gain"); assert ed == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert [VP.band(x, ed) for x in (-1, 0.2, 0.3, 0.79, 0.95)] == [0, 0, 1, 3, 4]
    cen = b.centres("A", "gain", ed); assert len(cen) == VP.N_BINS and cen == sorted(cen)
    assert VP.expected([0, 0, 1, 0, 0], cen) == pytest.approx(cen[2])
    for _ in range(200):
        b.add("A", "learning", 0.0)                                   # a constant part: every value in band 0, no crash
    e0 = b.edges("A", "learning"); assert VP.band(0.0, e0) == 0 and len(b.centres("A", "learning", e0)) == VP.N_BINS
    b.add("A", "gain", 5.0); assert len(b.vals[("A", "gain")]) == 100   # window keeps the newest values


def test_prompts_mix_and_composed():
    ed = [-0.4, -0.1, 0.1, 0.4]
    pp = VP.part_prompt("HEAD", "agency", "A", ed)
    assert pp.startswith("HEAD") and "A) below -0.40" in pp and "E) above +0.40" in pp and "choices change" in pp
    w = VP.scaled(VR.FUN_WEIGHTS_V2, 2.0); assert w["agency"] == pytest.approx(0.4)
    tp = VP.total_prompt("HEAD", "A", w, 0.3, {"gain": (3, ed), "agency": (0, ed)}, ed)
    assert "0.40 x agency" in tp and "- 0.30" in tp and "gain: band D (+0.10 to +0.40)" in tp and "agency: band A (below -0.40)" in tp
    assert "others" not in VP.mix_text(VR.WINNER_FUN_WEIGHTS_V2, 0.0)            # a zero weight is not shown
    parts = VP.parts_of(0.5, -0.3, {"agency": 1.0, "novelty": 0.5})
    assert set(parts) == set(VP.COMPONENTS) and parts["learning"] == 0.0
    assert VP.composed(parts, VR.FUN_WEIGHTS_V2, 0.1) == pytest.approx(0.5 - 0.3 + 0.2 * 1.5 - 0.1)
    note = VP.vote_note(2, ed, 0.05); assert "band C (-0.10 to +0.10)" in note and "+0.05" in note


def test_memory_note():
    m = VP.memory_note(1, [f"move{i}" for i in range(12)], 10, 14, [10, 10], [8, 10], True, trial=True)
    assert m.startswith("In a practice play") and "as P1" in m and "10 -> 14" in m and "(+4 more)" in m and "you won" in m
    assert "[10, 10] -> [8, 10]" in m and len(m) <= VP.MEMORY_CHARS
    assert VP.memory_note(0, [], 5, 5, [1], [1], None, trial=False).startswith("The last time you played")


def test_fun_share_and_spearman():
    assert VP.fun_share([(0.0, 1.0)]) is None
    assert VP.fun_share([(0.0, 0.0), (0.0, 1.0), (0.0, -1.0)]) == 0.0
    assert VP.fun_share([(1.0, 1.0), (-1.0, -1.0), (0.0, 0.0)]) == pytest.approx(0.5)
    assert VP.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0) and VP.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert VP.spearman([1, 1, 1], [1, 2, 3]) is None


# ------------------------------------------------------------------------------------------------------------- mixer
def test_fun_share_controller():
    k0 = VC.Controllers(dict(VC.PID)).knobs(); assert k0 == VC.fixed_knobs() and k0["fun_scale"] == 1.0   # off by default
    c = VC.Controllers(dict(VC.PID), ema=1.0, fun_share_lo=0.4)
    assert "fun" in c.pid and "fun" not in VC.CONSTRAINTS
    for _ in range(5):
        lam = c.update({"decline": 0.2, "harm": 0.1, "allow": 0.9, VC.FUN_STAT: 0.1})
    assert lam["fun"] > 0 and 1.0 < c.knobs()["fun_scale"] <= 1.0 + VC.LAMBDA_MAX
    c2 = VC.Controllers(dict(VC.PID), ema=1.0, fun_share_lo=0.4); c2.update({VC.FUN_STAT: 0.6})
    assert c2.knobs()["fun_scale"] == 1.0                                        # satisfied: no scaling
    c3 = VC.Controllers(dict(VC.PID), ema=1.0, fun_share_lo=0.4); c3.update({"decline": 0.2})
    assert c3.knobs()["fun_scale"] == 1.0                                        # not measured: hold


# ------------------------------------------------------------------------------------------------------------- wiring
def test_trainer_wiring():
    av = (PKG / "azr_vote.py").read_text()
    for k in ("from . import vote_predict as VP", 'prc = dict(cfg.get("predict") or {})', "fun_share_lo=cc.get(\"fun_share_lo\")",
              "def run_trials(tro)", "A3.play_batch3(server, pol, nets, inst, rng, gen, think=None)", "def make_predictions()",
              "VP.part_prompt(", "VP.total_prompt(", "extra=None if byst else player_extra(e, s, phase)", "remember(name, endow, bys_in, out, tr,",
              "d[\"targets\"] = [VP.band(v[part], d[\"edges\"])", "binner.add(role_of[key], part, x)", "(pred_w * ce.sum() / n_pd).backward()",
              'verdicts[i]["bonus_kind"] = "predicted"', "recent.append(it_)", "rng.choice(list(recent)) if from_recent", "VC.FUN_STAT: row.get(\"fun_share\")",
              "w_A_k = VP.scaled(", "weights=w_A_k", 'row["fun_share"] = VP.fun_share(fun_rows)', '"predictions.jsonl"', "offerings {cfg['offerings']} > worlds", 'pred_who = None if prc.get("adapter", "Q") == "own" else "cur:" + PRED_ROLE',
              '"who": pred_who or f"cur:{t_}"', 'float(cfg.get("lr_predictor", lr)) if r == PRED_ROLE'):
        assert k in av, k
    i = av.index("ctl.update(")
    assert av.index('row["fun_share"] =') < i                                     # the mixer reads this update's share
    assert av.index("n_trial = run_trials(tro)") < av.index("make_predictions(); tim") < av.index('run_votes("v1"); tim')
    assert av.index('d["targets"] =') > av.index("realised[(j, s)].append(pv)")    # targets only after the social rules
    assert av.index("(pred_w * ce.sum() / n_pd).backward()") < av.index("opt.step()")
    assert '"Q": "role_Q"' in (PKG / "azr3.py").read_text()
    for p in [PKG / "azr_vote.py", PKG / "vote_predict.py", PKG / "vote_control.py", PKG / "azr15_readout.py", ROOT / "scripts" / "check_azr3.py"]:
        compile(p.read_text(), str(p), "exec")
    ck = (ROOT / "scripts" / "check_azr3.py").read_text()
    assert "predict enabled but no prediction was ever made" in ck and "fun_share_lo set but no fun_share telemetry" in ck


# ------------------------------------------------------------------------------------------------------------- readout
def _run(d: Path, T=90, good=True):
    d.mkdir(parents=True)
    tel = [{"step": i, "decline_share_v1": 0.2, "played1": 0.9, "valid_share": 0.9, "valid_share_r0": 0.8, "rollbacks": 0,
            "pred_acc_total": (0.45 if i > 30 else 0.25) + 0.01 * (i % 3) if good else 0.2 + 0.01 * (i % 2),
            "fun_share": 0.5 if good else 0.1, "knob_fun_scale": 2.0, "recurring_offers": 3, "trials": 4} for i in range(T)]
    (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
    props, preds = [], []
    for s in range(T):
        for i in range(4):
            ag = (0.2 if s < T // 3 else 0.6) + 0.05 * i
            props.append({"step": s, "i": i, "reason": None, "code": f"g {s} {i}", "agency": ag, "vote_bonus": 0.1, "round": 0})
            real = 0.5 * ag + 0.01 * ((s + i) % 5) if good else 0.3 * ((s * 7 + i) % 3)
            preds.append({"step": s, "j": i, "seat": 0, "role": "A", "agency": ag, "exp_total": 0.4 * ag if good else 0.1 * ((s + i) % 2),
                          "realised": [{VP.TOTAL: real}]})
    (d / "proposals.jsonl").write_text("".join(json.dumps(p) + "\n" for p in props))
    (d / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    (d / "proposer_attempts.jsonl").write_text(""); (d / "traces.jsonl").write_text("")
    (d / "season_eval.json").write_text(json.dumps({"games": []}))


def test_readout(tmp_path):
    rep = {"config": {"readout_roles": {"main": "main"}, "reference_dir": str(tmp_path / "nope")}, "runs": {"main": {"spec": {"name": "main"}}}}
    _run(tmp_path / "a" / "runs" / "main")
    P = R15.readout(tmp_path / "a", rep)["predictions"]
    assert {"J1", "J2", "J3", "J4", "J5", "H1", "G1", "F1"} <= set(P)
    assert all(P[k]["held"] is True for k in ("J1", "J2", "J3", "J4", "J5")), {k: P[k]["held"] for k in ("J1", "J2", "J3", "J4", "J5")}
    _run(tmp_path / "b" / "runs" / "main", good=False)
    Pb = R15.readout(tmp_path / "b", rep)["predictions"]
    assert Pb["J1"]["held"] is False and Pb["J2"]["held"] is False and Pb["J3"]["held"] is False


# ------------------------------------------------------------------------------------------------------------- configs, scripts


