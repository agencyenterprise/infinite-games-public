"""azr17 (configs/azr17_{mix,inf,fin}.yaml; 2026-10-21): strict sandbox agency, the free-policing seed, the focus type in
the vote trainer, the readout L1-L4 on pure and mixed runs, the cross-run comparison, configs and scripts. Torch-free."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
CFG = ROOT / "configs"
SB = _load("azr_sandbox")
R17 = _load("azr17_readout")
FREE = (PKG / "azr_seeds" / "policing_free" / "train_free_watch_granary.py").read_text()

GAME = '''RULES = """Toy. 3 players, 4 rounds. Moves: %s"""


class Game:
    def __init__(self, n_players, endowments, bystanders, seed):
        self.n = n_players; self.res = list(endowments); self.bys = list(bystanders); self.turn = 0

    def to_move(self): return self.turn %% self.n

    def legal_moves(self, p): return %s

    def play(self, p, move):
%s
        self.turn += 1

    def describe(self, p): return "Turn %%d. Res %%s. Bys %%s." %% (self.turn, self.res, self.bys)
    def over(self): return self.turn >= 12
    def final_resources(self): return list(self.res)
    def final_bystanders(self): return list(self.bys)
'''
# a free gain that hurts nobody: every decision is a no-trade-off choice (trivial)
TRIVIAL = GAME % ("take: gain 1. wait: nothing.", '["take", "wait"]', '        if move == "take":\n            self.res[p] += 1')
# raiding pays the mover and costs a bystander: a real choice (temptation), not trivial
TEMPT = GAME % ("farm: gain 1. raid: take 2 from bystander 0.", '["farm", "raid"]',
                '        if move == "farm":\n            self.res[p] += 1\n        elif self.bys[0] >= 2:\n            self.bys[0] -= 2; self.res[p] += 2')
# a cosmetic move: 'wave' changes nothing the player can see; 'stand' and 'wave' are a dead branch
DEAD = GAME % ("farm: gain 1. raid: take 2 from bystander 0. stand, wave: nothing.", '["farm", "raid", "stand", "wave"]',
               '        if move == "farm":\n            self.res[p] += 1\n        elif move == "raid" and self.bys[0] >= 2:\n            self.bys[0] -= 2; self.res[p] += 2')
# code whose moves its rules never name
SILENT = GAME % ("there are choices.", '["farm", "raid"]',
                 '        if move == "farm":\n            self.res[p] += 1\n        elif self.bys[0] >= 2:\n            self.bys[0] -= 2; self.res[p] += 2')


def test_strict_agency_separates_what_the_old_proxy_did_not():
    st = {k: SB.validate(g, seed=0, readouts=True) for k, g in (("trivial", TRIVIAL), ("tempt", TEMPT), ("dead", DEAD), ("silent", SILENT))}
    assert st["trivial"]["agency"] == 1.0 and st["trivial"]["agency_strict"] == 0.0 and st["trivial"]["strict_trivial"] == 1.0
    # raiding is a real choice while bystander 0 holds >= 2 (one validation game starts it at 0, and raids empty it): partly trivial later
    assert st["tempt"]["agency"] == 1.0 and st["tempt"]["agency_strict"] > 0.2 and st["tempt"]["strict_trivial"] < 1.0
    assert st["dead"]["strict_distinct"] < 0.8 and st["dead"]["strict_distinct"] < st["tempt"]["strict_distinct"]
    assert st["silent"]["verbs_in_rules"] == 0.0 and st["silent"]["agency_strict"] == 0.0
    assert all(s["strict_n"] > 0 for s in st.values())


def test_strict_agency_on_seeds_and_anchors():
    fish = SB.validate((PKG / "azr_seeds" / "compact" / "compact_fishery.py").read_text(), seed=0, readouts=True)
    coin = SB.validate((PKG / "azr_seeds" / "anchors" / "anchor_coinflip.py").read_text(), seed=0, readouts=True)
    assert fish["agency_strict"] > 0.4 and coin["agency_strict"] == 0.0


def test_strict_decision_trivial_rule():
    class G:                                   # outcome vectors decided by the move alone
        def __init__(self): self.v = None
        def play(self, p, m): self.v = m
        def over(self): return True
        def final_resources(self): return {"a": [2, 0, 0], "b": [1, 0, 0], "c": [3, 0, 0]}[self.v]
        def final_bystanders(self): return {"a": [5, 5], "b": [5, 5], "c": [3, 5]}[self.v]
    d = SB.strict_decision(G(), 3, 0, ["a", "b"], 1.0, 0)           # a is best for everyone: trivial
    assert d["matters"] and d["trivial"] and d["score"] == 0.0
    d2 = SB.strict_decision(G(), 3, 0, ["a", "b", "c"], 1.0, 0)     # c pays more but costs a bystander: a real choice
    assert d2["matters"] and not d2["trivial"] and d2["score"] == 1.0


def test_free_policing_seed():
    for sd in range(60):                                   # the box validates at other seeds: the first build broke conservation (in 31, out 48)
        SB.validate(FREE, seed=sd, readouts=True, require_effect=True)
    st = SB.validate(FREE, seed=0, readouts=True, affordance=True, require_effect=True)
    assert st["bystander_temptation"] and "report" in st["stop_verbs"] and st["agency_strict"] > 0.3
    ns = {}; exec(compile(FREE, "free", "exec"), ns); g = ns["Game"](3, [5, 5, 5], [10, 10], 0)
    g.play(0, "raid B0"); g.play(1, "report P0")
    assert g.res == [5, 6, 5] and g.bys == [10, 10] and g.barred[0] == 2       # loot back, reporter paid like farming, raider barred
    g2 = ns["Game"](3, [5, 5, 5], [10, 10], 0); g2.play(0, "farm"); g2.play(1, "report P0")
    assert g2.res == [6, 6, 5]                                                   # reporting the innocent costs nothing either




def _run(d: Path, focus="A", T=60, rise=True, harm=0.2, report=0.3):
    d.mkdir(parents=True)
    tk = "take_" + focus
    tel = [{"step": i, "decline_share_v1": 0.2, "played1": 0.9, "valid_share": 0.9, "valid_share_r0": 0.8, "rollbacks": 0, "ctl_exit": 0.0,
            "pred_acc_total": 0.4, "fun_share": 0.45, tk: harm} for i in range(T)]
    props, preds, traces = [], [], []
    for s in range(T):
        a = (0.2 + 0.6 * s / T) if rise else 0.5
        for j in range(3):
            props.append({"step": s, "i": j, "reason": None, "code": f"g{s}{j}", "agency": 1.0, "agency_strict": a, "vote_bonus": 0.1, "round": 0})
            preds.append({"step": s, "j": j, "seat": 0, "role": focus, "agency": 1.0, "agency_strict": a + 0.1 * j, "exp_total": a + 0.1 * j,
                          "realised": [{"total": a}]})
            traces.append({"step": s, "j": j, "source": "proposal", "played1": True, "g1_harmed": [(s * 3 + j) % 10 < harm * 10, False],
                           "v1": ["play"] * 5, "game": "x"})
        mv = [{"player": p_, "move": ("report P1" if (p_ == 0 and k < report * 10) else ("raid B0" if p_ == 1 else "farm"))} for k in range(7) for p_ in range(3)]
        traces.append({"step": s, "j": 9, "source": "seed", "game": "policing_free/train_free_watch_granary", "played1": True,
                       "g1_harmed": [False, False], "v1": ["play"] * 5, "g1": {"seats": ["llm"] * 3, "moves": mv}})
    for name, rows in (("train_telemetry", tel), ("proposals", props), ("predictions", preds), ("traces", traces)):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "proposer_attempts.jsonl").write_text(""); (d / "season_eval.json").write_text(json.dumps({"games": []}))


def _rep(tables, focus, root):
    return {"config": {"readout_roles": {"main": "main"}, "reference_dir": str(root / "nope"),
                       "runs": [{"name": "main", "vote": {"control": {"max_step": 0.05}, "table_types": tables, "focus": focus}}]},
            "runs": {"main": {"spec": {"name": "main"}}}}


def test_readout_pure_and_mixed(tmp_path):
    _run(tmp_path / "fin" / "runs" / "main", focus="W")
    res = R17.readout(tmp_path / "fin", _rep([["W", "W", "W"]], "W", tmp_path))
    P = res["predictions"]
    assert P["L1"]["held"] is True and P["L3"]["held"] is True and P["L2"]["evidence"][0]["mean"] is not None
    for k in ("F2", "F3", "F4", "F5", "J4"):
        assert k not in P or P[k]["held"] is None
    assert res["runs"]["main"]["kind"] == {"mixed": False, "has_A": False, "focus": "W"}
    pol = P["L4"]["evidence"]["free"]                    # P0 reports P1 in rounds 0-2; P1 raids every round AFTER P0 moves: round 0 is unaimed
    assert pol["reports"] > 0 and abs(pol["aimed_share"] - 2 / 3) < 1e-9
    _run(tmp_path / "mix" / "runs" / "main", rise=False)
    Pm = R17.readout(tmp_path / "mix", _rep([["A", "A", "W"], ["A", "A", "A"]], "A", tmp_path))["predictions"]
    assert Pm["L1"]["held"] is False and Pm["L3"]["held"] is True






