"""azr14 (configs/azr14_fun.yaml; 2026-10-18): agency + outcome-uncertainty fun terms, winners, the degenerate guard, the
voter-only proposer bonus, KL to the frozen warm proposer, rollback, the readout H1-H4, configs and scripts. Torch-free."""
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
PB = _load("proposer_bench")
R14 = _load("azr14_readout")


# ------------------------------------------------------------------------------------------------------------- fun terms
def test_defaults_unchanged_and_v2_terms():
    assert VR.FUN_TERMS == ("learning", "novelty", "surprise", "empowerment", "others") and set(VR.FUN_WEIGHTS) == set(VR.FUN_TERMS)
    assert set(VR.FUN_WEIGHTS_V2) == set(VR.FUN_TERMS) | {"agency", "uncertainty"} and VR.WINNER_FUN_WEIGHTS_V2["others"] == 0.0
    fs = VR.FunState()
    t1 = VR.own_terms(fs, "g", "A", 0.5, [3, 3], ["s"], True); assert "agency" not in t1                      # night-8 call unchanged
    t2 = VR.own_terms(fs, "g", "A", 0.5, [3, 3], ["s"], True, agency=1.4); assert t2["agency"] == 1.0 and t2["uncertainty"] == 1.0
    base = {k: 0.0 for k in VR.FUN_TERMS if k != "others"}
    r_old, _ = VR.seat_rewards([0.2, 0.1], [dict(base), dict(base)])
    r_new, _ = VR.seat_rewards([0.2, 0.1], [dict(base, agency=1.0, uncertainty=0.0), dict(base, agency=0.0, uncertainty=0.0)], weights=VR.FUN_WEIGHTS_V2)
    assert r_new[0] > r_old[0] and r_new[0] > r_new[1]                              # a game where my choices matter is more fun
    assert VR.fun_total(dict(base, others=0.5, agency=1.0, uncertainty=1.0), VR.FUN_WEIGHTS_V2) == pytest.approx(0.2 * 2.5)


def test_uncertainty_peaks_at_even_odds():
    fs = VR.FunState(); assert fs.uncertainty("g", "A") == 1.0                      # never played: open
    for _ in range(200):
        fs.update_win("g", "A", True)
    assert fs.uncertainty("g", "A") < 0.01                                         # always winning is dull
    for _ in range(200):
        fs.update_win("h", "A", False)
    assert fs.uncertainty("h", "A") < 0.01                                         # always losing too
    for i in range(400):
        fs.update_win("k", "A", i % 2 == 0)
    assert fs.uncertainty("k", "A") > 0.95


def test_winners():
    assert VR.winners([0.5, 0.2, 0.1], [True, True, True]) == [True, False, False]
    assert VR.winners([0.5, 0.5, 0.1], [True, True, True]) == [False, False, False]          # a tie is no win
    assert VR.winners([0.1, 0.9, 0.2], [True, False, True]) == [False, None, True]


# ------------------------------------------------------------------------------------------------------------- proposer
def test_degenerate_guard():
    game = (PKG / "azr_seeds" / "anchors" / "anchor_coinflip.py").read_text()
    assert not PB.degenerate(game) and not PB.degenerate("")
    assert PB.degenerate(game + "\n" + "[\n" * 30)                                    # azr13's late proposals
    assert PB.degenerate("x = " + "]" * 400)
    assert PB.degenerate("\n".join(["a = 1", "b = 2"] * 40))
    assert PB.proposer_reward({"ok": False, "reason": "degenerate"}, 9000, {"no_code_reward": -1.5, "len_penalty_valid_only": True}) == -1.5


def test_trainer_wiring():
    av = (PKG / "azr_vote.py").read_text()
    for k in ('vote_all = bool(pc.get("vote_all", False))', 'fun_v2 = fun and bool(vc.get("fun_v2", False))', "VR.FUN_WEIGHTS_V2 if fun_v2",
              'REF_ROLE = "R"', 'ref_who = "cur:" + REF_ROLE', "nets.use(ref_who)", "degen = [p_degen and PB.degenerate(t) for t in txt_]",
              '"reason": "degenerate"', "def vote_probe(", "pv = vote_probe(", "for i in ([] if vote_all else valid):",
              'q.copy_(rb_state[n_].to(q.device, q.dtype)); opt.state.pop(q, None)', '"best_proposer"', 'or step == rb.get("force_step")',
              "fs.update_win(name, t_, bool(won[s]))", '"agency": verdicts[i]["stats"].get("agency")', '"rollbacks": n_rollback',
              "agency=(0.0 if ag is None else ag) if fun_v2 else None"):
        assert k in av, k
    assert av.index("pv = vote_probe(") < av.index("5. update")                      # probes run before the model goes to train mode
    assert av.index("opt.step()") < av.index("rb_hist.append(")
    a3 = (PKG / "azr3.py").read_text(); assert '"R": "role_R"' in a3
    pipe = (PKG / "azr3_pipeline.py").read_text()
    assert 'if nets.roles and "R" in nets.roles:' in pipe and "ref_tensors=n_copied" in pipe and '"stats": {"agency": v["stats"].get("agency"),' in pipe   # azr17: + agency_strict
    for p in [PKG / "azr_vote.py", PKG / "vote_rules.py", PKG / "azr3.py", PKG / "azr3_pipeline.py", PKG / "proposer_bench.py",
              PKG / "azr14_readout.py", ROOT / "scripts" / "check_azr3.py"]:
        compile(p.read_text(), str(p), "exec")


# ------------------------------------------------------------------------------------------------------------- readout
def _run(d: Path, T=120, collapse=False):
    d.mkdir(parents=True)
    tel = [{"step": i, "decline_share_v1": 0.2, "played1": 0.9, "valid_share": (0.1 if collapse and 60 <= i < 110 else 0.9),
            "valid_share_r0": 0.8, "rollbacks": 1 if i > 80 else 0, "fun_agency": 0.5} for i in range(T)]
    (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
    props = []
    for s in range(T):
        for i in range(6):
            ag = (0.1 if s < T // 3 else 0.6) + 0.05 * i
            props.append({"step": s, "i": i, "reason": None, "code": f"g {s} {i}", "agency": ag, "vote_bonus": 0.3 + 0.5 * ag, "round": 0})
    (d / "proposals.jsonl").write_text("".join(json.dumps(p) + "\n" for p in props))
    (d / "proposer_attempts.jsonl").write_text(""); (d / "traces.jsonl").write_text("")
    (d / "season_eval.json").write_text(json.dumps({"games": []}))


def test_readout(tmp_path):
    _run(tmp_path / "a" / "runs" / "main")
    rep = {"config": {"readout_roles": {"main": "main"}, "reference_dir": str(tmp_path / "nope")}, "runs": {"main": {"spec": {"name": "main"}}}}
    P = R14.readout(tmp_path / "a", rep)["predictions"]
    assert {"H1", "H2", "H3", "H4", "G1", "F1"} <= set(P)
    assert P["H1"]["held"] is True and P["H2"]["held"] is True and P["H3"]["held"] is True and P["H1"]["evidence"]["rollbacks"] == 1
    _run(tmp_path / "b" / "runs" / "main", collapse=True)
    assert R14.readout(tmp_path / "b", rep)["predictions"]["H1"]["held"] is False


# ------------------------------------------------------------------------------------------------------------- configs, scripts


