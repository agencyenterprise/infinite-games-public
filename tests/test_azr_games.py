"""Unit tests for the AZR games line: the sandbox (whitelist, validation, live server), the economy and rewards, the
proposer/solver text helpers, the readout's predictions, every seed game, and the wiring of the pipeline. The torch
modules are checked by AST only; the process-based tests skip where the package cannot be imported in a child
process (the Mac, where src/__init__ imports torch)."""
from __future__ import annotations

import ast
import copy
import glob
import importlib
import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
SEEDS = sorted(glob.glob(str(PKG / "azr_seeds" / "*.py")))
COMPACT = sorted(glob.glob(str(PKG / "azr_seeds" / "compact" / "*.py")))
CONFIGS = [ROOT / "configs" / "azr_games.yaml", ROOT / "configs" / "azr_games_smoke.yaml"]
SCRIPTS = [ROOT / "scripts" / "azr_games_smoke.sh", ROOT / "scripts" / "azr_games.sh"]


def _load(name: str):
    try:
        return importlib.import_module(f"infinite_games.{name}")
    except ImportError:
        pass
    if "src" not in sys.modules:
        pkg = types.ModuleType("src"); pkg.__path__ = [str(ROOT / "src")]; sys.modules["src"] = pkg
    if "infinite_games" not in sys.modules:
        sub = types.ModuleType("infinite_games"); sub.__path__ = [str(PKG)]; sys.modules["infinite_games"] = sub
    spec = importlib.util.spec_from_file_location(f"infinite_games.{name}", PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod); return mod


SB = _load("azr_sandbox")
EC = _load("azr_economy")
TX = _load("azr_text")
AR = _load("azr_readout")
ORCHARD = (PKG / "azr_seeds" / "train_orchard.py").read_text()


def _child_import_ok() -> bool:
    try:
        importlib.import_module("infinite_games.azr_sandbox"); import torch  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------------- sandbox
BAD = {
    "import": "import os\nRULES='x'\nclass Game: pass\n",
    "from_import": "from subprocess import run\nRULES='x'\nclass Game: pass\n",
    "dunder_attr": "RULES='x'\nclass Game:\n    def __init__(s):\n        ().__class__\n",
    "format_ref": "RULES='x'\nf=''.format\nclass Game: pass\n",
    "eval": "RULES='x'\neval('1')\nclass Game: pass\n",
    "getattr": "RULES='x'\ngetattr(1,'real')\nclass Game: pass\n",
    "open": "RULES='x'\nopen('/etc/passwd')\nclass Game: pass\n",
    "global": "RULES='x'\ndef f():\n    global y\nclass Game: pass\n",
    "no_game": "RULES='x'\n",
    "no_rules": "class Game: pass\n",
    "missing_method": "RULES='x'\nclass Game:\n    def __init__(self, n, e, b, s): pass\n",
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_static_check_rejects(name):
    with pytest.raises(SB.SandboxError):
        SB.static_check(BAD[name])


def test_orchard_validates_and_reports_structure():
    st = SB.validate(ORCHARD, 0)
    assert st["turns"] > 0 and 0.0 <= st["agency"] <= 1.0 and st["all_players_move"] and st["distinct_moves"] >= 3
    assert isinstance(st["bystander_temptation"], bool) and isinstance(st["ruin_temptation"], bool)


def _mutant(old, new):
    assert ORCHARD.count(old) == 1, old
    return ORCHARD.replace(old, new)


@pytest.mark.parametrize("code,reason", [
    (lambda: _mutant("self.rng = random.Random(seed)", "self.rng = random.Random()").replace("take = min(2, self.stock)", "take = min(self.rng.randint(1, 2), self.stock)"), "nondeterministic"),
    (lambda: _mutant("return self.round >= self.rounds", "return False"), "too_long_game"),
    (lambda: _mutant("take = min(3, self.res[q]); self.res[q] -= take; self.res[p] += take", "self.res[p] += 50"), "conservation"),
    (lambda: _mutant("return list(self.res)", "return [-1] * self.n"), "payout"),
    (lambda: _mutant('moves = ["harvest"]', "moves = []\n        return moves"), "legal_moves"),
    (lambda: _mutant("return self.turn % self.n", "return 7"), "to_move"),
    (lambda: _mutant("self.turn += 1", "self.turn += 1\n        while True:\n            pass"), "timeout"),
])
def test_validation_names_the_failure(code, reason):
    with pytest.raises(SB.SandboxError) as e:
        SB.validate(code(), 0)
    assert e.value.reason == reason


def test_format_is_rewritten_and_safe():
    ok = ORCHARD.replace('rows = ["Round %d of %d. You are P%d." % (self.round + 1, self.rounds, p)]',
                         'rows = ["Round {} of {}. You are P{}.".format(self.round + 1, self.rounds, p)]')
    assert ok != ORCHARD and SB.validate(ok, 0)["turns"] > 0
    assert SB.safe_format("P{} has {:>3}", 1, 5) == "P1 has   5"
    for bad in ("{0.__class__}", "{0[0]}", "{:{0.real}}", "{x.y}"):
        with pytest.raises(SB.SandboxError):
            SB.safe_format(bad, 1, x=1)
    esc = ORCHARD.replace('return "\\n".join(rows)', 'return "{0.__class__}".format(self)')
    assert esc != ORCHARD
    with pytest.raises(SB.SandboxError):
        SB.validate(esc, 0)


def test_games_must_last_min_turns():
    short = _mutant("return self.round >= self.rounds", "return True")
    with pytest.raises(SB.SandboxError) as e:
        SB.validate(short, 0)
    assert e.value.reason == "too_short_game"


def test_conservation_bound():
    assert SB.conservation_ok([10, 10, 10], [10, 10], {"resources": [20, 20, 20], "bystanders": [7, 8]})
    assert not SB.conservation_ok([10, 10, 10], [10, 10], {"resources": [40, 20, 20], "bystanders": [0, 0]})


@pytest.mark.skipif(not _child_import_ok(), reason="child processes cannot import the package here (no torch)")
def test_game_server_plays_to_the_end_and_pool_validates():
    srv = SB.GameServer(2); rng = random.Random(0)
    try:
        ob = srv.step({0: {"op": "new", "code": ORCHARD, "n": 3, "endowments": [10, 10, 10], "bystanders": [10, 10], "seed": 3}})[0]
        turns = 0
        while not ob["over"]:
            ob = srv.step({0: {"op": "play", "player": ob["player"], "move": rng.choice(ob["moves"])}})[0]; turns += 1
        assert turns > 0 and SB.conservation_ok([10, 10, 10], [10, 10], ob)
        bad = srv.step({1: {"op": "new", "code": BAD["import"], "n": 3, "endowments": [1, 1, 1], "bystanders": [1, 1], "seed": 0}})[1]
        assert "error" in bad
    finally:
        srv.close()
    pool = SB.Pool(2)
    try:
        v = pool.validate_many([ORCHARD, BAD["eval"]])
        assert v[0]["ok"] and not v[1]["ok"] and v[1]["reason"] == "name"
    finally:
        pool.close()


# ------------------------------------------------------------------------------------------------------ seed games
@pytest.mark.parametrize("path", SEEDS + COMPACT, ids=[Path(p).stem for p in SEEDS + COMPACT])
def test_every_seed_game_validates_and_tempts(path):
    code = Path(path).read_text()
    for seed in (0, 1):
        st = SB.validate(code, seed)
        assert st["all_players_move"] and st["agency"] > 0
    low = code.lower()
    assert "knock out" not in low and "eliminat" not in low


def test_seed_split():
    names = [Path(p).name for p in SEEDS]
    assert any(n.startswith("train_") for n in names) and any(n.startswith("heldout_") for n in names)
    assert all(n.startswith(("train_", "heldout_")) for n in names)


# ---------------------------------------------------------------------------------------------------------- economy
def test_settle_carries_resources_and_restarts_on_ruin():
    w = EC.World(); e = w.endowments()
    ev = EC.settle(w, e, {"resources": [15, 10, 5], "bystanders": [10, 10]})
    assert not ev["restarted"] and w.wallets == [15, 10, 5]
    ev = EC.settle(w, w.endowments(), {"resources": [30, 0, 0], "bystanders": [10, 10]})
    assert ev["ruin"] and ev["restarted"] and w.wallets == [EC.START_WALLET] * 3 and w.restarts == 1


STOMP_OUT = {"resources": [12, 12, 12], "bystanders": [2, 10]}


def test_stomp_restores_the_bystander_and_keeps_going():
    w = EC.World(wallets=[10, 10, 10]); ev = EC.settle(w, w.endowments(), STOMP_OUT, restore_prob=1.0, bitter_games=3)
    assert ev["stomp"] and ev["restored"] and not ev["restarted"] and w.bystanders == [EC.START_BYSTANDER, 10]
    assert w.wallets == [12, 12, 12] and w.bitter_left == 3 and w.restores == 1 and w.stomps == 1 and w.restarts == 0
    ev = EC.settle(w, w.endowments(), {"resources": [12, 12, 12], "bystanders": [10, 10]}, restore_prob=1.0, bitter_games=3)
    assert ev["stomped_before"] and not ev["stomp"] and w.bitter_left == 2


def test_stomp_can_restart_the_world_instead():
    w = EC.World(wallets=[10, 10, 10]); ev = EC.settle(w, w.endowments(), STOMP_OUT, restore_prob=0.0, bitter_games=3)
    assert ev["stomp"] and ev["restarted"] and not ev["restored"] and w.bitter_left == 0 and w.wallets == [EC.START_WALLET] * 3
    assert ev["bitter_games"] == 3                       # the forward charge is the same whichever way the draw goes


def test_restore_draw_is_seeded_and_mixes():
    def branch(seed, n=200):
        w = EC.World(seed=seed); out = []
        for _ in range(n):
            out.append(EC.settle(w, w.endowments(), dict(STOMP_OUT), restore_prob=0.5)["restored"]); w.wallets = [10, 10, 10]
        return out
    a, b = branch(3), branch(3)
    assert a == b and 0.35 < sum(a) / len(a) < 0.65 and branch(4) != a


def test_ruin_restarts_even_with_a_stomp():
    w = EC.World(); ev = EC.settle(w, w.endowments(), {"resources": [30, 0, 0], "bystanders": [0, 10]}, restore_prob=1.0)
    assert ev["ruin"] and ev["restarted"] and not ev["restored"]


def test_game_outcome_matches_settle_without_advancing():
    w = EC.World(); snap = EC.World(**vars(w))
    a = EC.game_outcome(snap, STOMP_OUT, 3); b = EC.settle(w, snap.endowments(), STOMP_OUT, restore_prob=1.0, bitter_games=3)
    keys = ("stomp", "bitter_games", "stomped_before", "ruin")
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys} and snap.games == 0 and snap.bitter_left == 0


def test_bitter_forward_charge_and_discount():
    base = {"gains": [10, -5, -5], "ruin": False, "bystander_loss": 0.0, "top_share": 0.4}
    fresh = EC.rewards("infinite", [10, 10, 10], dict(base, stomp=True, bitter_games=3, stomped_before=False))
    assert fresh == pytest.approx([g - (EC.STOMP_PENALTY + 3 * EC.BITTER_PER_GAME) for g in (1.0, -0.5, -0.5)])
    bitter = EC.rewards("infinite", [10, 10, 10], dict(base, stomp=False, bitter_games=0, stomped_before=True))
    assert bitter == pytest.approx([1.0 * (1 - EC.BITTER_DISCOUNT), -0.5, -0.5])      # gains taste bitter, losses do not
    assert EC.rewards("finite", [10, 10, 10], dict(base, stomp=False, bitter_games=0, stomped_before=True)) == [1.0, -0.5, -0.5]
    off = {k: 0.0 for k in EC.DEFAULT_WEIGHTS}
    assert EC.rewards("infinite", [10, 10, 10], dict(base, stomp=True, bitter_games=3, stomped_before=True), off) == [1.0, -0.5, -0.5]


def test_every_weight_is_switched_by_a_condition():
    assert set(EC.WEIGHT_CONDITION) == set(EC.DEFAULT_WEIGHTS) and set(EC.WEIGHT_CONDITION.values()) <= set(EC.CONDITIONS)


def test_no_carryover_clears_the_bitterness():
    w = EC.World(); EC.settle(w, w.endowments(), STOMP_OUT, carryover=False, restore_prob=1.0)
    assert w.bitter_left == 0 and w.bystanders == [EC.START_BYSTANDER] * 2 and w.wallets == [EC.START_WALLET] * 3


def test_entry_cap_keeps_the_rest_outside():
    w = EC.World(wallets=[100, 10, 10]); e = w.endowments()
    assert e == [EC.ENTRY_CAP, 10, 10]
    EC.settle(w, e, {"resources": [EC.ENTRY_CAP, 10, 10], "bystanders": [10, 10]})
    assert w.wallets == [100, 10, 10]


def test_no_carryover_resets_every_game():
    w = EC.World(); EC.settle(w, w.endowments(), {"resources": [20, 5, 5], "bystanders": [9, 9]}, carryover=False)
    assert w.wallets == [EC.START_WALLET] * 3 and w.bystanders == [EC.START_BYSTANDER] * 2


def test_rewards_by_arm_and_condition():
    ev = {"gains": [10, -5, -5], "ruin": True, "stomp": False, "bystander_loss": 0.5, "top_share": 0.8, "bitter_games": 0, "stomped_before": False}
    fin = EC.rewards("finite", [10, 10, 10], ev); inf = EC.rewards("infinite", [10, 10, 10], ev)
    assert fin == [1.0, -0.5, -0.5]
    pen = EC.RUIN_PENALTY + EC.BYSTANDER_LOSS_PENALTY * 0.5 + EC.MONOPOLY_PENALTY * (0.8 - EC.MONOPOLY_SHARE)
    assert inf == pytest.approx([f - pen for f in fin])
    off = EC.rewards("infinite", [10, 10, 10], ev, {"ruin": 0.0, "stomp": 1.0, "bitter": 1.0, "bystander_loss": 0.0, "monopoly": 0.0})
    assert off == fin
    with pytest.raises(ValueError):
        EC.rewards("nope", [1, 1, 1], ev)


def test_stage_rewards_climb_with_progress():
    R = EC.STAGE_REWARD
    assert R["no_code"] < R["syntax"] < R["interface"] < R["exec"] < R["runtime"] < R["runtime_in_play"] < 0
    assert EC.proposer_reward({"ok": False, "reason": "unheard_of"}, 0.7) == EC.STAGE_DEFAULT


def test_learnability_shape():
    assert EC.learnability(0.5) == 0 and EC.learnability(1.0) == 0 and EC.learnability(0.3) == 0
    assert EC.learnability(0.75) == pytest.approx(1.0) and 0 < EC.learnability(0.6) < 1


def test_proposer_reward_cases_and_switches():
    st = {"agency": 0.5, "all_players_move": True, "bystander_temptation": True, "ruin_temptation": False}
    v = {"ok": True, "stats": st}
    assert EC.proposer_reward({"ok": False, "reason": "no_code"}, 0.7) == -1.0
    assert EC.proposer_reward({"ok": False, "reason": "syntax"}, 0.7) == EC.STAGE_REWARD["syntax"]
    assert EC.proposer_reward(v, 0.75) == pytest.approx(1.0 * 0.5 * 0.5)
    assert EC.proposer_reward(v, 0.75, use_temptation=False) == pytest.approx(0.5)
    assert EC.proposer_reward(v, 0.2, use_learnability=False, use_agency=False, use_temptation=False) == 1.0
    assert EC.proposer_reward(dict(v, stats=dict(st, all_players_move=False)), 0.75) == 0.0


def test_gini():
    assert EC.gini([5, 5, 5]) == pytest.approx(0) and EC.gini([0, 0, 9]) == pytest.approx(2 / 3) and EC.gini([]) == 0


def test_validation_covers_a_ruined_bystander():
    src = (PKG / "azr_sandbox.py").read_text()
    assert "bys[0] = 0" in src


# ------------------------------------------------------------------------------------------------------------- text
def test_extract_code():
    assert TX.extract_code("blah\n```python\nRULES='x'\nclass Game: pass\n```\nmore") == "RULES='x'\nclass Game: pass\n"
    assert TX.extract_code("RULES = 'a'\nclass Game:\n  pass") is not None
    assert TX.extract_code("no code here") is None
    assert TX.extract_code("Here:\n```python\nRULES='x'\nclass Game:\n    def over(self):\n        return Tr") .startswith("RULES")


def test_prompts():
    p = TX.proposer_prompt([ORCHARD]); assert "class Game" in p and "temptation" in p and SB.INTERFACE_TEXT in p
    s = TX.solver_prompt("rules", "state", ["a", "b", "c"], 1); assert "C) c" in s and "player P1" in s
    s = TX.solver_prompt("rules", "state", ["a", "b"], 2, "WORLD TEXT", ["P0 (turn 0): hi"], "my plan")
    assert "WORLD TEXT" in s and "P0 (turn 0): hi" in s and "my plan" in s and s.endswith("Reply with the letter of your choice.")
    r = TX.reflect_prompt("rules", "state", ["raid B0", "plant"], 0, "WORLD", [])
    assert "THINK:" in r and "SAY:" in r and "- raid B0" in r and "(none yet)" in r and "A)" not in r


def test_restrained_moves_and_thought_consistency():
    assert TX.restrained_moves(["raid P1", "plant", "glean B0", "harvest"]) == ["plant", "harvest"]
    assert TX.restrained_moves(["raid P1", "take 2 from B0"]) == ["raid P1", "take 2 from B0"]
    assert TX.acts_on_thought("plant", "I will plant a tree") and not TX.acts_on_thought("raid P1", "I will plant")
    assert TX.novice_seats(2, TX.RESTRAINED) == [TX.LLM, TX.LLM, TX.RESTRAINED]


def test_compact_seeds_are_compact():
    assert COMPACT, "no compact seed games"
    for p in COMPACT:
        assert len(Path(p).read_text()) <= 2600, p


def test_rules_of():
    assert TX.rules_of(ORCHARD).startswith("Orchard.") and TX.rules_of("x = 1") is None and TX.rules_of(None) is None
    assert TX.rules_of("RULES = \'\'\'single quotes\'\'\'\nclass Game: pass") == "single quotes"


def test_parse_reflection():
    r = TX.parse_reflection("THINK: raiding B0 would restart the world.\nSAY: Let's trade fairly.\nmore")
    assert r == {"think": "raiding B0 would restart the world.", "say": "Let's trade fairly."}
    assert TX.parse_reflection("no markers at all") == {"think": "no markers at all", "say": ""}
    assert TX.parse_reflection("THINK: x\nSAY: nothing")["say"] == ""
    assert len(TX.parse_reflection("THINK: y\nSAY: " + "z" * 1000)["say"]) == TX.MAX_SAY_CHARS


def test_world_text_states_facts_not_rewards():
    w = EC.World(wallets=[12, 5, 13]); t = EC.world_text(w)
    assert "P1 5" in t and "restarts" in t and "Recent games" not in t
    EC.settle(w, w.endowments(), {"resources": [12, 5, 13], "bystanders": [10, 10]})
    assert "Recent games" in EC.world_text(w) and "most: P2" in EC.world_text(w)
    assert "standalone" in EC.world_text(None) and "starts fresh" in EC.world_text(EC.World(), carryover=False)
    low = t.lower()
    assert "penal" not in low and "reward" not in low                  # the same text in every arm
    for _ in range(6):
        EC.settle(w, w.endowments(), {"resources": list(w.endowments()), "bystanders": [10, 10]})
    assert len(w.history) == EC.HISTORY_GAMES
    assert TX.novice_seats(4).count(TX.NOVICE) == 1 and TX.novice_seats(4)[1] == TX.NOVICE


# ---------------------------------------------------------------------------------------------------------- readout
def _season(collapse, ruin=0.0, stomp=0.0, top=0.5, n=40, comp=0.8):
    rng = random.Random(0)
    return {"games": {(0, s, 0): {"collapse": collapse + rng.uniform(-.01, .01), "ruin": ruin, "stomp": stomp, "bystander_loss": 0.1,
                                  "top_share": top, "welfare": 50, "gain": 0.0} for s in range(n)}, "competence": [comp] * 10}


def test_predictions_q3_q5_q7():
    runs = {n: {"spec": {}} for n in ("f", "i", "nr", "nb", "nc")}
    S = {"f": _season(0.6, ruin=0.5), "i": _season(0.2, ruin=0.1, stomp=0.1), "nr": _season(0.5, ruin=0.4, stomp=0.1),
         "nb": _season(0.5, ruin=0.1, stomp=0.4), "nc": _season(0.4)}
    res = {"runs": {n: {"train": {"steps": 0}, "proposer": {}} for n in runs}}
    P = AR.predictions(res, S, {n: {} for n in runs}, {"finite": "f", "infinite": "i", "no_ruin": "nr", "no_bystander": "nb", "no_carryover": "nc"}, runs)
    assert P["Q3"]["held"] is True and P["Q5"]["held"] is True and P["Q7"]["held"] is True
    S2 = dict(S, i=_season(0.7))
    assert AR.predictions(res, S2, {n: {} for n in runs}, {"finite": "f", "infinite": "i"}, runs)["Q3"]["held"] is False


# ----------------------------------------------------------------------------------------------------------- wiring
def _leaf_keys(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaf_keys(v, prefix + k + ".")
        yield prefix + k






def _defs(path: Path):
    tree = ast.parse(path.read_text()); out = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for c in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        init = next((f for f in c.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
        if init is not None:
            init = copy.deepcopy(init); init.args.args = init.args.args[1:]; out[c.name] = init
    return out


def _imported_names(path: Path, module: str):
    tree = ast.parse(path.read_text()); names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.endswith(module):
            names |= {a.asname or a.name for a in n.names}
    return names


def _check_calls(caller: Path, alias: str, targets):
    defs = {}
    for t in targets:
        defs.update(_defs(t))
    n_calls = 0
    for node in ast.walk(ast.parse(caller.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == alias:
            fn = defs.get(node.func.attr)
            if fn is None:
                continue
            params = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
            n_req = len(fn.args.args) - len(fn.args.defaults)
            assert len(node.args) <= len(fn.args.args), f"{alias}.{node.func.attr}: too many positional args"
            for kw in node.keywords:
                assert kw.arg is None or kw.arg in params, f"{alias}.{node.func.attr}: unknown keyword {kw.arg}"
            assert set(params[:n_req]) <= set(params[:len(node.args)]) | {kw.arg for kw in node.keywords}, f"{alias}.{node.func.attr}: missing args"
            n_calls += 1
    return n_calls






def _code_lines(p: Path):
    return [l for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]




def test_entry_point_indexed_and_no_salted_hash():
    assert 'run_main("azr_games", main)' in (PKG / "azr_pipeline.py").read_text()
    for p in sorted(PKG.glob("azr*.py")):
        tree = ast.parse(p.read_text())
        assert not any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "hash" for x in ast.walk(tree)), p.name
