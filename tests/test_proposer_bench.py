"""Proposer bench (src/infinite_games/proposer_bench.py; 2026-10-15): prompt variants, the import fix, verdicts with the effect rule,
repair chats, the summary, configs and scripts. Torch-free."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
CFG = ROOT / "configs"
PB = _load("proposer_bench")
SB = _load("azr_sandbox")
TX = _load("azr_text")
GAME = (PKG / "azr_seeds" / "anchors" / "anchor_coinflip.py").read_text()


def test_prompt_variants():
    cur = PB.prompt_text(["A", "B"], "current"); same = PB.prompt_text(["A", "B"], "same_size")
    assert cur == TX.proposer_prompt3(["A", "B"]) and PB.CURRENT_ASK in cur
    assert PB.CURRENT_ASK not in same and PB.SAME_SIZE_ASK in same and same.replace(PB.SAME_SIZE_ASK, PB.CURRENT_ASK) == cur
    with pytest.raises(ValueError):
        PB.prompt_text(["A"], "nope")


def test_fix_imports_makes_a_forgotten_import_valid():
    assert "import random" in GAME and PB.fix_imports(GAME) == GAME                       # nothing missing: unchanged
    broken = GAME.replace("import random\n", "")
    assert "random." in broken and "import random" not in broken
    with pytest.raises(SB.SandboxError) as e:
        SB.validate(broken, seed=1)
    assert e.value.reason == "init"
    fixed = PB.fix_imports(broken); assert fixed.count("import random") == 1
    assert fixed.index("import random") > fixed.index('RULES')                            # placed after RULES
    SB.validate(fixed, seed=1)
    assert PB.fix_imports("x = math.sqrt(4)\n").startswith("import math\n")              # no RULES: at the top


def test_verdict_of():
    v = PB.verdict_of({"ok": True, "stats": {"effect": 0.9}}, SB.EFFECT_MIN); assert v["ok"] and v["ok_effect"] and v["reason"] is None
    v = PB.verdict_of({"ok": True, "stats": {"effect": 0.0}}, SB.EFFECT_MIN); assert v["ok"] and not v["ok_effect"] and v["reason"] == "no_effect"
    v = PB.verdict_of({"ok": False, "reason": "syntax", "detail": "x" * 5000}, SB.EFFECT_MIN)
    assert not v["ok"] and v["reason"] == "syntax" and len(v["detail"]) == PB.DETAIL_CHARS


def test_repair_messages():
    m = PB.repair_messages("sys", "ask", [{"text": "t1", "reason": "init", "detail": "NameError"}, {"text": "t2", "reason": "runtime", "detail": ""}])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]                 # only the last attempt is shown
    assert m[2]["content"] == "t2" and "runtime: (no detail)" in m[3]["content"]
    m1 = PB.repair_messages("sys", "ask", [{"text": "t1", "reason": "init", "detail": "NameError"}]); assert "init: NameError" in m1[3]["content"]


def test_wilson_and_summarise():
    w = PB.wilson(5, 10); assert w["lo"] < 0.5 < w["hi"] and PB.wilson(0, 0)["p"] is None
    rows = [{"pid": 0, "round": 0, "ok": False, "ok_effect": False, "reason": "init", "chars": 100, "cjk": False, "code": "a"},
            {"pid": 0, "round": 1, "ok": True, "ok_effect": True, "reason": None, "chars": 100, "cjk": False, "code": "a b"},
            {"pid": 1, "round": 0, "ok": True, "ok_effect": False, "reason": "no_effect", "chars": 0, "cjk": True, "code": "c"},
            {"pid": 1, "round": 1, "ok": False, "ok_effect": False, "reason": "syntax", "chars": 50, "cjk": False, "code": "d"}]
    s = PB.summarise(rows)
    assert s["n_proposals"] == 2 and s["rounds"]["0"]["ok_effect"]["k"] == 0 and s["rounds"]["1"]["ok_effect"]["k"] == 1
    assert s["rounds"]["0"]["ok"]["k"] == 1 and s["rounds"]["1"]["ok"]["k"] == 2                 # cumulative over rounds
    assert s["first_reasons"] == {"init": 1, "no_effect": 1} and s["cjk_first"] == 0.5 and s["no_code_first"] == 0.5 and s["distinct_valid"] == 1








def test_seed_games_and_warm_examples():
    import random as _r
    seeds = PB.seed_games(PKG / "azr_seeds", [".", "night3", "sanction_full_guilty"])
    assert len(seeds) == 29 and not any(s["name"].split("/")[-1].startswith("heldout_") for s in seeds)
    ex = PB.warm_examples(seeds[:5], seeds, 20, 2, 3200, "same_size", _r.Random(0), 6000)
    assert len(ex) == 20 and all(e["target"].startswith("```python\n") and e["target"].endswith("\n```") for e in ex)
    for e in ex:                                                      # the target game is never one of its own examples
        tgt = e["target"][len("```python\n"):-len("\n```")]
        assert tgt.strip() not in e["prompt"]
    assert PB.warm_examples([], seeds, 5, 2, 3200, "current", _r.Random(0), 6000) == []
