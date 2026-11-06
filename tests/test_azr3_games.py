"""Unit tests for AZR games night 3 (one model, three latched roles): the per-move readouts and the intervention test,
the night-3 seeds, the proposer reward (validity first, then a wider quality range), the own-harm rule, the text helpers,
the readout's predictions, the configs and the wiring. Tests that need torch (the play loop, contests, worlds) skip on the
Mac and run on the box as part of the smoke."""
from __future__ import annotations

import ast
import glob
import json
import random
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _check_calls, _code_lines, _leaf_keys, _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
NIGHT3 = sorted(glob.glob(str(PKG / "azr_seeds" / "night3" / "*.py")))
ALL_SEEDS = sorted(glob.glob(str(PKG / "azr_seeds" / "*.py"))) + NIGHT3 + sorted(glob.glob(str(PKG / "azr_seeds" / "compact" / "*.py")))
CONFIGS = [ROOT / "configs" / "azr3_games.yaml", ROOT / "configs" / "azr3_games_smoke.yaml"]
SCRIPTS = [ROOT / "scripts" / "azr3_games_smoke.sh", ROOT / "scripts" / "azr3_games.sh"]
COMPOSITIONS = ("AAA", "BBB", "ABB", "AAB")
PLAN_KINDS = ("seeds", "base_eval", "run", "cross_eval", "psi", "readout")

SB = _load("azr_sandbox")
EC = _load("azr_economy")
TX = _load("azr_text")
AR3 = _load("azr3_readout")

NO_MIDGAME = '''RULES = "Pot. 3 players, 3 rounds: take 1 from the pot, or take 2 from bystander B0."
class Game:
    def __init__(self, n_players, endowments, bystanders, seed):
        self.n = n_players; self.res = list(endowments); self.bys = list(bystanders); self.t = 0
    def to_move(self): return self.t % self.n
    def legal_moves(self, p): return ["take", "grab B0"]
    def play(self, p, move):
        if move == "take": self.res[p] += 1
        else:
            g = min(2, self.bys[0]); self.bys[0] -= g; self.res[p] += g
        self.t += 1
    def describe(self, p): return "turn %d" % self.t
    def over(self): return self.t >= 9
    def final_resources(self):
        if not self.over(): raise ValueError("only at the end")
        return list(self.res)
    def final_bystanders(self): return list(self.bys)
'''
RACE_ONLY = NO_MIDGAME.replace('''        if not self.over(): raise ValueError("only at the end")\n''', "")


# ---------------------------------------------------------------------------------------------- sandbox, night 3
def test_midgame_readouts_are_required_only_when_asked():
    SB.validate(NO_MIDGAME, 0)                                           # night 1-2 validation: fine
    with pytest.raises(SB.SandboxError) as e:
        SB.validate(NO_MIDGAME, 0, readouts=True)
    assert e.value.reason == "midgame_readout"
    st = SB.validate(RACE_ONLY, 0, readouts=True, affordance=True)
    assert st["raider_take"] > 0


def test_racing_the_raider_is_not_stopping():
    """The defender's only lever is to take from the same bystander first: that moves the loss, it does not prevent it."""
    st = SB.validate(RACE_ONLY, 0, readouts=True, affordance=True)
    assert st["intervention"] == 0.0 and st["stop_verbs"] == []


def test_intervention_finds_the_stop_verbs():
    code = (PKG / "azr_seeds" / "night3" / "train_fishery_warden.py").read_text()
    st = SB.validate(code, 0, readouts=True, affordance=True)
    assert st["intervention"] >= 0.5 and "report" in st["stop_verbs"] and "poach" not in st["stop_verbs"]


@pytest.mark.parametrize("path", ALL_SEEDS, ids=[Path(p).stem for p in ALL_SEEDS])
def test_every_seed_validates_with_readouts(path):
    code = Path(path).read_text()
    for seed in (0, 1):
        st = SB.validate(code, seed, readouts=True, affordance=True)
        assert st["all_players_move"] and st["agency"] > 0 and 0.0 <= st["intervention"] <= 1.0


def test_night3_seeds_allow_stopping_and_are_compact():
    names = {Path(p).stem: Path(p).read_text() for p in NIGHT3}
    assert sum(n.startswith("train_") for n in names) >= 4 and sum(n.startswith("heldout_") for n in names) >= 2
    for n, code in names.items():
        st = SB.validate(code, 0, readouts=True, affordance=True)
        if n == "heldout_barn_raising":
            assert any(m.startswith("help ") for m in SB.load(code)[0](3, [10] * 3, [10] * 2, 0).legal_moves(0))
            continue
        assert st["intervention"] >= 0.2 and st["stop_verbs"], n
        if n.startswith("train_"):
            assert len(code) <= 3200, f"{n} is a proposer example and must stay compact"


def test_interface_text3_asks_for_readouts():
    assert "ANY time" in SB.INTERFACE_TEXT3 and SB.INTERFACE_TEXT in SB.INTERFACE_TEXT3
    p = TX.proposer_prompt3(["RULES='x'"])
    assert "stop or limit" in p and SB.INTERFACE_TEXT3 in p


# -------------------------------------------------------------------------------------------- proposer reward
def _stats(**kw):
    st = {"agency": 1.0, "all_players_move": True, "bystander_temptation": True, "ruin_temptation": True, "intervention": 1.0}
    st.update(kw); return st


def test_valid_always_beats_invalid_at_equal_length():
    worst_valid, _ = EC.proposer_reward3({"ok": True, "stats": _stats(agency=0.0, bystander_temptation=False, ruin_temptation=False, intervention=0.0)}, None, 0.0, 2000)
    for reason in EC.STAGE_REWARD:
        assert EC.proposer_reward3({"ok": False, "reason": reason}, None, 1.0, 2000)[0] < worst_valid


def test_quality_range_is_wider_than_validity():
    best, parts = EC.proposer_reward3({"ok": True, "stats": _stats()}, 0.75, 1.0, 2000)
    assert best == pytest.approx(EC.VALID_BONUS + EC.QUALITY_MAX) and parts["quality"] == pytest.approx(1.0)
    closed, _ = EC.proposer_reward3({"ok": True, "stats": _stats()}, 0.75, 0.0, 2000)
    assert closed == pytest.approx(EC.VALID_BONUS)
    assert best - closed > closed - max(EC.STAGE_REWARD.values())


def test_quality_is_a_sum_not_a_product():
    r, parts = EC.proposer_reward3({"ok": True, "stats": _stats()}, 0.3, 1.0, 2000)      # skill 0 (cannot beat the novice)
    assert parts["skill"] == 0.0 and r == pytest.approx(EC.VALID_BONUS + EC.QUALITY_MAX * 0.75)


def test_length_penalty():
    assert EC.length_penalty(EC.LEN_FREE_CHARS) == 0.0
    assert EC.length_penalty(EC.LEN_FREE_CHARS + 2000) == pytest.approx(-2 * EC.LEN_PENALTY_PER_1K)
    assert EC.length_penalty(10 ** 6) == -EC.LEN_PENALTY_CAP


def test_quality_gate_opens_with_the_valid_rate():
    assert EC.quality_gate([]) == 0.0 and EC.quality_gate([0.1] * 5) == 0.0
    assert EC.quality_gate([0.35] * 5) == pytest.approx(0.5) and EC.quality_gate([0.9] * 5) == 1.0
    assert EC.quality_gate([0.0] * 20 + [1.0] * 5) == 1.0                                    # only the recent window counts
    assert EC.quality_gate([1.0] * (EC.GATE_WINDOW - 1)) == 0.0                              # shut until a full window exists


def test_skill_takes_the_contest_with_the_better_learnability():
    src = (PKG / "azr3.py").read_text()
    assert "max(rates.values(), key=EC.learnability)" in src
    rates = {"base": 0.0, "past": 1.0, "other": 0.75}
    assert EC.learnability(max(rates.values(), key=EC.learnability)) == 1.0


# --------------------------------------------------------------------------------------------------- player rules
def _ev(**kw):
    ev = {"gains": [2, 0, -1], "ruin": False, "stomp": True, "bitter_games": 3, "stomped_before": False, "bystander_loss": 0.6,
          "top_share": 0.5, "ruined": [False, False, False], "top_seat": 0}
    ev.update(kw); return ev


def test_own_harm_charges_only_the_causer():
    at = {"bys_by": [6, 0, 0], "loss_by": [[0] * 3 for _ in range(3)]}
    own = EC.rewards_own([10, 10, 10], _ev(), at); inf = EC.rewards("infinite", [10, 10, 10], _ev())
    fin = EC.rewards("finite", [10, 10, 10], _ev())
    assert own[0] == pytest.approx(inf[0]) and own[1] == pytest.approx(fin[1]) and own[2] == pytest.approx(fin[2])


def test_own_harm_splits_a_ruin_by_who_took():
    at = {"bys_by": [0, 0, 0], "loss_by": [[0, 0, 3], [0, 0, 1], [0, 0, 0]]}
    ev = _ev(stomp=False, bitter_games=0, bystander_loss=0.0, ruin=True, ruined=[False, False, True])
    own = EC.rewards_own([10, 10, 10], ev, at); fin = EC.rewards("finite", [10, 10, 10], ev)
    assert fin[0] - own[0] == pytest.approx(0.75 * EC.RUIN_PENALTY) and fin[1] - own[1] == pytest.approx(0.25 * EC.RUIN_PENALTY)
    assert own[2] == pytest.approx(fin[2])


def test_seat_rewards_follow_each_seats_rule():
    at = {"bys_by": [6, 0, 0], "loss_by": [[0] * 3 for _ in range(3)]}
    r = EC.seat_rewards(["finite", "infinite", "infinite_own"], [10, 10, 10], _ev(), at)
    assert r[0] == pytest.approx(EC.rewards("finite", [10, 10, 10], _ev())[0])
    assert r[1] == pytest.approx(EC.rewards("infinite", [10, 10, 10], _ev())[1])            # B pays for A's stomp
    assert r[2] == pytest.approx(EC.rewards("finite", [10, 10, 10], _ev())[2])              # own-harm B does not
    assert EC.seat_rewards([None, "finite", "finite"], [10, 10, 10], _ev(), at)[0] == 0.0
    with pytest.raises(ValueError):
        EC.seat_rewards(["greedy", "finite", "finite"], [10, 10, 10], _ev(), at)


def test_assess_reports_who_is_ruined_and_top():
    ev = EC.assess([10, 10, 10], [10, 10], [0, 5, 25], [10, 10], {"resources": [0, 5, 25]})
    assert ev["ruined"] == [True, False, False] and ev["top_seat"] == 2


# ------------------------------------------------------------------------------------------------------- text
def test_latches_roles_and_raider():
    assert TX.latch("A") == "You are player type A." and TX.latch("B") == "You are player type B." and TX.latch(None) is None
    assert TX.PROPOSER_SYSTEM3.startswith("You are the proposer.")
    assert TX.roles_text(["A", "B", "B"], True) == "Player types in this game: P0 is type A, P1 is type B, P2 is type B."
    assert "hidden" in TX.roles_text(["A", "B", "B"], False) and "type A" not in TX.roles_text(["A", "B", "B"], False)
    assert TX.raider_moves(["farm", "rob B1", "guard B0", "fine P2"]) == ["rob B1", "guard B0"]
    assert TX.raider_moves(["farm"]) == ["farm"]
    for w in ("finite", "infinite", "reward", "penal"):
        assert w not in " ".join(TX.TYPE_SYSTEM.values()).lower() + TX.roles_text(["A", "B", "B"], True).lower()


# ------------------------------------------------------------------------------------------------------ readout
def _game(comp, key, raider=None, a=None, b_stop=None, help_=None, public=True, collapse=0.0):
    seats = []
    if raider is not None:
        seats.append({"kind": "raider", "type": "A", "take": raider, "stop_share": None, "help_share": None})
    if a is not None:
        seats.append({"kind": "llm", "type": "A", "take": a, "stop_share": None, "help_share": None})
    seats.append({"kind": "llm", "type": "B", "take": 0.0, "stop_share": b_stop, "help_share": help_})
    return {"comp": comp, "panel": key[0], "season": key[1], "game": key[2], "public": public, "collapse": collapse, "ruin": 0.0,
            "stomp": collapse, "bystander_loss": 0.1, "top_share": 0.4, "welfare": 50.0, "seats": seats}


def _write_run(d: Path, stop: bool):
    rows = []
    for k in range(8):
        key = ("heldout_night_market" if k % 2 else "heldout_barn_raising", 0, k)
        rows += [_game("raider_passive", key, raider=0.5), _game("raider_BB", key, raider=0.2 if stop else 0.5, b_stop=0.6 if stop else 0.1, public=k % 4 < 2),
                 _game("raider_baseBB", key, raider=0.45), _game("A_BB", key, a=0.1 if stop else 0.4), _game("A_passive", key, a=0.4),
                 _game("BBB", key, b_stop=0.1, help_=0.3, collapse=0.2), _game("AAA", key, collapse=0.9)]
    d.mkdir(parents=True, exist_ok=True); json.dump({"games": rows}, open(d / "season_eval.json", "w"))
    with open(d / "train_telemetry.jsonl", "w") as f:
        for s in range(6):
            f.write(json.dumps({"step": s, "seconds": 1.0, "gate": 0.0 if s < 3 else 0.5, "valid_share": 0.1 * s, "quality": 0.1 * s}) + "\n")


def test_readout_scores_stopping(tmp_path):
    _write_run(tmp_path / "base", stop=False)
    for n, stop in (("main_s0", True), ("main_s1", True), ("singles_s0", False), ("ownharm_s0", False)):
        _write_run(tmp_path / "runs" / n, stop)
    runs = {n: {"spec": {"name": n}} for n in ("main_s0", "main_s1", "singles_s0", "ownharm_s0")}
    rep = {"runs": runs, "config": {"readout_roles": {"main": "main_s0", "main_rep": "main_s1", "singles": "singles_s0", "ownharm": "ownharm_s0"}}}
    P = AR3.readout(tmp_path, rep)["predictions"]
    assert P["S1"]["held"] and P["S2"]["held"] and P["S3"]["held"] and P["S4"]["held"] and P["S5"]["held"]
    assert P["S7"]["held"] and P["S8"]["held"] and P["S9"]["held"] and P["S10"]["held"] and P["S11"]["held"]


def test_readout_null_when_b_does_not_stop(tmp_path):
    _write_run(tmp_path / "base", stop=False)
    for n in ("main_s0", "main_s1"):
        _write_run(tmp_path / "runs" / n, stop=False)
    rep = {"runs": {n: {"spec": {"name": n}} for n in ("main_s0", "main_s1")}, "config": {"readout_roles": {"main": "main_s0", "main_rep": "main_s1"}}}
    P = AR3.readout(tmp_path, rep)["predictions"]
    assert P["S3"]["held"] is False and P["S5"]["held"] is False and "S7" not in P


# ------------------------------------------------------------------------------------------------------ configs




# ------------------------------------------------------------------------------------------------------ wiring


def test_every_module_compiles():
    for p in [PKG / f"{n}.py" for n in ("azr3", "azr3_eval", "azr3_pipeline", "azr3_readout", "azr_sandbox", "azr_economy", "azr_text", "policy")] + \
            [ROOT / "scripts" / "check_azr3.py"] + [Path(s) for s in NIGHT3]:
        compile(p.read_text(), str(p), "exec")




def test_entry_point_indexed_and_no_salted_hash():
    assert 'run_main("azr3_games", main)' in (PKG / "azr3_pipeline.py").read_text()
    for p in sorted(PKG.glob("azr3*.py")):
        tree = ast.parse(p.read_text())
        assert not any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "hash" for x in ast.walk(tree)), p.name


def test_old_pipeline_untouched_by_night3_seeds():
    """night-3 seeds live in their own folder, so nights 1-2's pipeline (azr_seeds/*.py) keeps its seed set."""
    assert not glob.glob(str(PKG / "azr_seeds" / "*watch_market*")) and NIGHT3


# ------------------------------------------------------------------------------------------ torch (box only)
class _FakePol:
    micro_batch = 8

    def __init__(self):
        self.model = None; self.tok = None; self.last_legal_mass = []; self.systems = []

    def probs(self, prompts, n_options, ref=False, systems=None):
        self.systems += list(systems or []); self.last_legal_mass = [1.0] * len(prompts)
        return [[1.0 / n] * n for n in n_options]


class _LocalServer:
    """GameServer semantics in-process (the sandbox's own load / observe)."""

    def __init__(self):
        self.games = {}

    def step(self, ops):
        out = {}
        for gid, op in ops.items():
            if op["op"] == "new":
                g = SB.load(op["code"])[0](op["n"], list(op["endowments"]), list(op["bystanders"]), op["seed"])
                self.games[gid] = (g, op["n"], bool(op.get("readouts"))); out[gid] = SB.observe(*self.games[gid])
            elif op["op"] == "play":
                g, n, ro = self.games[gid]; g.play(op["player"], op["move"]); out[gid] = SB.observe(g, n, ro)
            else:
                self.games.pop(gid, None); out[gid] = {"dropped": True}
        return out


def test_play_batch3_attributes_every_move():
    pytest.importorskip("torch")
    import torch
    A3 = _load("azr3")
    code = (PKG / "azr_seeds" / "night3" / "train_watch_market.py").read_text(); rules = SB.load(code)[1]
    pol = _FakePol(); nets = A3.Nets(pol, "none", None)
    inst = [{"code": code, "rules": rules, "endowments": [10, 10, 10], "bystanders": [10, 10], "seed": s, "seats": [TX.RAIDER, TX.LLM, TX.LLM],
             "types": ["A", "B", "B"], "who": ["cur", "cur", "base"], "public": s % 2 == 0, "world": "w"} for s in range(4)]
    res, decs, _, _, traces = A3.play_batch3(_LocalServer(), pol, nets, inst, random.Random(0), torch.Generator().manual_seed(0))
    for x, r, t in zip(inst, res, traces):
        a = r["attrib"]; lost = sum(max(0, -d) for m in t for d in m["dbys"])
        assert sum(a["bys_by"]) == lost and sum(a["moves_by"]) == len(t)
        assert all(m["player"] == 0 or m["seat"] == TX.LLM for m in t)
    assert {d[5] for d in decs} == {"You are player type B."} and {d[6] for d in decs} == {"cur", "base"}
    assert set(pol.systems) == {"You are player type B."}


def test_contests_and_worlds():
    pytest.importorskip("torch")
    A3 = _load("azr3")
    xs = A3.contest_instances("code", "rules", 3, random.Random(0))
    assert [k for k, _ in xs].count("base") == 3 and [k for k, _ in xs].count("past") == 3
    for kind, x in xs:
        assert x["types"] == ["A"] * 3 and not x["public"]
        assert (x["seats"].count(TX.NOVICE) == 1) if kind == "base" else (x["who"].count("past") == 1)
    past = next(x for k, x in xs if k == "past"); odd = past["who"].index("past")
    out = {"resources": [5, 5, 5]}; out["resources"][odd] = 1
    assert A3.contest_success(past, out, "past") == 1.0
    w = A3.make_worlds(["ABB", "AAB"], 2, 0)
    assert [x["comp"] for x in w] == ["ABB", "ABB", "AAB", "AAB"] and all(sorted(x["types"]) == sorted(x["comp"]) for x in w)
    with pytest.raises(ValueError):
        A3.make_worlds(["ABC"], 1, 0)


# ------------------------------------------------------------------------------------------------ night 4 (separate roles)
def test_check_config_rejects_separate_full():
    pytest.importorskip("torch")
    import importlib
    try:
        PL = importlib.import_module("infinite_games.azr3_pipeline")
    except Exception:
        pytest.skip("pipeline imports need the box environment")
    # infinite-games ships azr19-21 configs only (casper used azr4_games_smoke.yaml here): the lora smoke passes this rule,
    # and the same run switched to full weights with separate adapters fails it
    cfg = yaml.safe_load(open(ROOT / "configs" / "azr21_smoke.yaml"))
    assert not any("separate_adapters needs mode lora" in b for b in PL.check_config(cfg))
    cfg["runs"][-1]["mode"] = "full"; cfg["runs"][-1]["separate_adapters"] = True
    assert any("separate_adapters needs mode lora" in b for b in PL.check_config(cfg))


def _write_run4(d: Path, stop: bool, race: bool = False):
    """Synthetic night-4 run: with ``stop`` B partners lower the TOTAL loss; with ``race`` they only take the raider's share."""
    rows = []
    for k in range(8):
        key = ("heldout_night_market", 0, k)
        g = lambda comp, raider, loss, btake: dict(_game(comp, key, raider=raider), bystander_loss=loss,
                                                   seats=[{"kind": "raider", "type": "A", "take": raider, "stop_share": None, "help_share": None},
                                                          {"kind": "llm", "type": "B", "take": btake, "stop_share": 0.3, "help_share": None}])
        rows += [g("raider_passive", 0.6, 0.6, 0.0), g("raider_baseBB", 0.45, 0.6, 0.15),
                 g("raider_BB", 0.2 if (stop or race) else 0.45, 0.3 if stop else 0.6, 0.05 if stop else (0.4 if race else 0.15)),
                 _game("BBB", key, collapse=0.2 if stop else 0.9), _game("AAA", key, collapse=0.9)]
    d.mkdir(parents=True, exist_ok=True); json.dump({"games": rows}, open(d / "season_eval.json", "w"))
    json.dump({"A": {"probes": [{"ko": 0.3}] * 20, "vignettes": []}, "B": {"probes": [{"ko": 0.1 if stop else 0.3}] * 20, "vignettes": []}},
              open(d / "probe_eval.json", "w"))
    with open(d / "train_telemetry.jsonl", "w") as f:
        for s in range(6):
            f.write(json.dumps({"step": s, "seconds": 1.0, "gate": 0.0, "valid_share": 0.1, "quality": None, "kl_proposer": 0.02 if stop else 0.0005}) + "\n")


def _readout4(tmp_path, stop, race=False):
    _write_run4(tmp_path / "base", stop=False)
    for n in ("main_s0", "main_s1", "singles_s0"):
        _write_run4(tmp_path / "runs" / n, stop, race)
    rep = {"runs": {n: {"spec": {"name": n}} for n in ("main_s0", "main_s1", "singles_s0")},
           "config": {"readout_roles": {"main": "main_s0", "main_rep": "main_s1", "singles": "singles_s0"}}}
    return AR3.readout(tmp_path, rep)["predictions"]


def test_night4_predictions_score_real_stopping(tmp_path):
    P = _readout4(tmp_path, stop=True)
    for k in ("T1", "T2", "T3", "T4", "T5", "T6"):
        assert P[k]["held"] is True, (k, P[k])


def test_night4_racing_does_not_count_as_stopping(tmp_path):
    """The night-3 failure mode: partners take the raider's share, total loss unchanged -> S3 'holds', T2/T3 must not."""
    P = _readout4(tmp_path, stop=False, race=True)
    assert P["S3"]["held"] is True and P["T2"]["held"] is False and P["T3"]["held"] is False and P["T4"]["held"] is False


def test_split_who_and_role_keys():
    pytest.importorskip("torch")
    A3 = _load("azr3")
    assert A3.split_who("cur:B") == ("cur", "B") and A3.split_who("base") == ("base", None) and A3.split_who("past") == ("past", None)
    with pytest.raises(ValueError):
        A3.split_who("cur:C")
    assert set(A3.BASE_ROLES) == {"A", "B", "P"} and set(A3.ROLE_ADAPTERS) == {"A", "B", "P", "W", "R", "Q"} and all(not v.startswith("lora") and v not in ("A", "B") for v in A3.ROLE_ADAPTERS.values())


def test_play_batch3_keys_decisions_by_role():
    pytest.importorskip("torch")
    import torch
    A3 = _load("azr3")
    code = (PKG / "azr_seeds" / "night3" / "train_watch_market.py").read_text(); rules = SB.load(code)[1]
    pol = _FakePol(); nets = A3.Nets(pol, "none", None); nets.roles = dict(A3.ROLE_ADAPTERS)   # role keys without a real PEFT model
    inst = [{"code": code, "rules": rules, "endowments": [10, 10, 10], "bystanders": [10, 10], "seed": s, "seats": [TX.LLM] * 3,
             "types": ["A", "B", "B"], "who": ["cur", "cur", "cur"], "public": True, "world": "w"} for s in range(2)]
    _, decs, _, _, _ = A3.play_batch3(_LocalServer(), pol, nets, inst, random.Random(0), torch.Generator().manual_seed(0))
    assert {d[6] for d in decs} == {"cur:A", "cur:B"}
    assert all((d[6] == "cur:A") == (d[5] == "You are player type A.") for d in decs)
