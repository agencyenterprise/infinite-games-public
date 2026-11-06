"""azr13 (scripts/azr13.sh; 2026-10-16): bench -> registered selection -> warm start + repaired proposer in azr12's game. The selection rule,
the proposer reward options, the trainer and pipeline wiring, the readout G1-G3, templates and scripts. Torch-free."""
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
PB = _load("proposer_bench")
EC = _load("azr_economy")
R13 = _load("azr13_readout")


def _arm(name, p, lo, repairs=0, temp=0.7, ok=None):
    return {"arm": {"name": name, "temperature": temp, "prompt": "current", "fix_imports": True, "repairs": repairs},
            "rounds": {str(repairs): {"ok_effect": {"p": p, "lo": lo, "hi": p + 0.05}, "ok": {"p": p if ok is None else ok, "lo": lo, "hi": p}}}}


def test_choose_prefers_small_model_unless_clearly_beaten():
    s = {"models": {"m3": {"a": _arm("a", 0.10, 0.06), "b": _arm("b", 0.20, 0.14, repairs=2)},
                    "m7": {"a": _arm("a", 0.25, 0.18, repairs=2)}}}
    c = PB.choose(s, ["m3", "m7"]); assert c["model"] == "m3" and c["arm"] == "b"           # 0.25 < 1.5 x 0.20
    s["models"]["m7"]["a"] = _arm("a", 0.35, 0.28, repairs=2)
    c = PB.choose(s, ["m3", "m7"]); assert c["model"] == "m7" and c["settings"]["repairs"] == 2 # 0.35 >= 0.30 and 0.28 > 0.20
    s["models"]["m7"]["a"] = _arm("a", 0.35, 0.19, repairs=2)
    assert PB.choose(s, ["m3", "m7"])["model"] == "m3"                                       # lower bound not above the small one's score


def test_choose_ties_and_fallbacks():
    s = {"models": {"m3": {"x": _arm("x", 0.102, 0.05, repairs=2), "y": _arm("y", 0.100, 0.05, repairs=0, temp=1.0)}}}
    c = PB.choose(s, ["m3", "m7"]); assert c["arm"] == "y" and c["model"] == "m3"            # within TIE: fewer repairs; missing model skipped
    z = {"models": {"m3": {"x": _arm("x", 0.0, 0.0, ok=0.05), "y": _arm("y", 0.0, 0.0, ok=0.01)}}}
    c = PB.choose(z, ["m3"]); assert c["key"] == "ok" and c["arm"] == "x"                     # nothing passes the effect rule: fall back
    with pytest.raises(ValueError):
        PB.choose({"models": {}}, ["m3"])


def test_proposer_reward_options():
    long_fail = {"ok": False, "reason": "syntax"}; ch = 7000
    plain = PB.proposer_reward(long_fail, ch, {}); fixed = PB.proposer_reward(long_fail, ch, {"len_penalty_valid_only": True})
    assert plain == pytest.approx(EC.STAGE_REWARD["syntax"] + EC.length_penalty(ch)) and fixed == pytest.approx(EC.STAGE_REWARD["syntax"])
    empty = PB.proposer_reward({"ok": False, "reason": "no_code"}, 0, {"len_penalty_valid_only": True, "no_code_reward": -1.5})
    assert empty == -1.5 and empty < min(EC.STAGE_REWARD.values())                           # empty is the worst outcome, whatever its length
    ok = PB.proposer_reward({"ok": True, "stats": {"agency": 0.5, "all_players_move": True, "bystander_temptation": True, "ruin_temptation": False}}, ch,
                            {"len_penalty_valid_only": True})
    assert ok > fixed and ok < EC.VALID_BONUS                                                 # valid games still pay the length penalty




def _run(d: Path, T=30):
    d.mkdir(parents=True)
    tel = [{"step": i, "decline_share_v1": 0.2, "played1": 0.9, "valid_share_r0": 0.05 if i < 10 else 0.15} for i in range(T)]
    (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
    props = []
    for s in range(T):
        for i in range(8):
            ok = (i < 1) if s < 10 else (i < 3)
            props.append({"step": s, "i": i, "reason": None if ok else "runtime", "code": f"game {s} {i}" if ok else None, "round": 0})
    (d / "proposals.jsonl").write_text("".join(json.dumps(p) + "\n" for p in props))
    (d / "proposer_attempts.jsonl").write_text("".join(json.dumps({"step": 0, "i": i, "round": r, "ok": r == 1}) + "\n" for i in range(4) for r in (0, 1)))
    for f in ("traces.jsonl",):
        (d / f).write_text("")
    (d / "season_eval.json").write_text(json.dumps({"games": []}))


def test_readout(tmp_path):
    _run(tmp_path / "runs" / "main")
    rep = {"config": {"readout_roles": {"main": "main"}, "reference_dir": str(tmp_path / "nope")},
           "runs": {"main": {"spec": {"name": "main"}, "warm": {"nll_first": 1.0, "nll_last": 0.5, "examples": 8}}}}
    out = R13.readout(tmp_path, rep); P = out["predictions"]
    assert {"G1", "G2", "G3", "F1", "F6"} <= set(P)
    assert P["G1"]["held"] is True and P["G2"]["held"] is True and P["G3"]["held"] is True
    assert out["runs"]["main"]["repair_rounds"][1]["ok"] == 1.0 and out["runs"]["main"]["warm"]["examples"] == 8


