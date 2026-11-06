"""Sanctions in the sanction-night games (azr_seeds/sanction/, written by scripts/make_sanction_games.py).

Pure python (no torch), shared by training (azr3.train_run3), evaluation (azr3_eval.season_eval3) and the readout.

    player_verbs     the verbs a game's rules let one player aim at another ("- <verb> P<k>: ..."); in the sanction games this
                     is exactly the sanction, and night-3/4 games' fine / report count too
    is_control_game  the control twin: its sanction "is noted, and nothing else happens"
    rule_of          which guilt rule a game uses: "last" (the original games: the target's most recent move took from a
                     bystander) or "any" (the 2026-10-03 regimes: the target took from a bystander this game and has not
                     been sanctioned since); read from the rules text, so every caller agrees with the game
    sanction_events  per sanction move in a trace: who sanctioned whom, and whether it was JUSTIFIED (the target's most recent
                     move took from a bystander, read from the per-move bystander deltas, so it holds for every seat kind)
    sanction_credit  the optional shaped reward (the "credit" run only): ``per`` for each justified sanction a seat plays,
                     capped at ``cap`` per game; unjustified sanctions earn nothing
    seat_stats       per seat: share of its moves that are sanctions, justified sanctions, and sanctions aimed at a type-A seat
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

PLAYER_VERB_RE = re.compile(r"^- (\w+) P<k>", re.M)
TARGET_RE = re.compile(r"^P(\d)$")
CONTROL_MARK = "nothing else happens"
GUILT_ANY_MARK = "is guilty if it has taken"
RULES = ("last", "any")


def player_verbs(rules: str) -> List[str]:
    return sorted(set(PLAYER_VERB_RE.findall(rules or "")))


def is_control_game(rules: str) -> bool:
    return CONTROL_MARK in (rules or "")


def rule_of(rules: str) -> str:
    return "any" if GUILT_ANY_MARK in (rules or "") else "last"


def _target(move: str, verbs: Sequence[str]) -> Optional[int]:
    w = move.split()
    if len(w) == 2 and w[0] in verbs:
        m = TARGET_RE.match(w[1])
        return int(m.group(1)) if m else None
    return None


def sanction_events(trace: Sequence[Dict], verbs: Sequence[str], types: Sequence[Optional[str]], rule: str = "last") -> List[Dict]:
    """``trace``: the game's moves in order, each with player, move and dbys (per-bystander deltas of that move). ``rule``: the
    game's guilt rule (rule_of)."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r} (valid: {RULES})")
    last: Dict[int, bool] = {}; owing: Dict[int, bool] = {}; out = []
    for m in trace:
        p = m["player"]; q = _target(m["move"], verbs) if verbs else None
        if q is not None:
            guilty = last.get(q, False) if rule == "last" else owing.get(q, False)
            out.append({"seat": p, "type": types[p] if p < len(types) else None, "target": q,
                        "target_type": types[q] if q < len(types) else None, "justified": bool(guilty)})
            owing[q] = False
        stole = any(d < 0 for d in (m.get("dbys") or [])); last[p] = stole
        if stole:
            owing[p] = True
    return out


def sanction_credit(trace: Sequence[Dict], verbs: Sequence[str], n: int, per: float, cap: float, rule: str = "last") -> List[float]:
    out = [0.0] * n
    for e in sanction_events(trace, verbs, [None] * n, rule):
        if e["justified"]:
            out[e["seat"]] += per
    return [min(cap, c) for c in out]


def seat_stats(trace: Sequence[Dict], verbs: Sequence[str], types: Sequence[Optional[str]], seat: int, rule: str = "last") -> Dict[str, Optional[float]]:
    mv = sum(1 for m in trace if m["player"] == seat)
    if not mv:
        return {"sanction_share": None, "sanction_just": None, "sanction_at_A": None, "sanction_at_W": None}
    ev = [e for e in sanction_events(trace, verbs, types, rule) if e["seat"] == seat]
    return {"sanction_share": len(ev) / mv, "sanction_just": sum(e["justified"] for e in ev) / mv,
            "sanction_at_A": sum(e["target_type"] == "A" for e in ev) / mv, "sanction_at_W": sum(e["target_type"] == "W" for e in ev) / mv}
