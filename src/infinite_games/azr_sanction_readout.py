"""Readout for the sanction night (configs/azr5_sanction.yaml). Pure python over the run directory.

The question (Thomas 2026-10-02): night 4's infinite player B learned to keep its hands off the bystanders but never to stop a
player who takes from them. Was that because the games barely offered a sanction (16-19% of training games, and B's sanction
rate FELL over training)? Tonight every game offers one: the EFFECTIVE games' sanction makes the last thief hand back what it
took and bars it for two rounds; each has a CONTROL twin whose sanction costs the same and does nothing. No proposer.

Roles (config ``readout_roles``): plain, plain_rep (the plain infinite reward, two seeds), credit (B also earns a bonus per
justified sanction). Every comparison is paired over (panel game, season, game) unless noted; "justified" = the target's most
recent move took from a bystander (azr_sanction.sanction_events). Registered predictions:
    U1  plain, both seeds: next to the scripted raider in the EFFECTIVE games, trained B plays more justified sanctions than
        untrained B (raider_BB vs raider_baseBB)
    U2  plain, both seeds: that increase is larger in the effective games than in their control twins (unpaired across the
        two sets of paired differences) -- B sanctions because it works, not because it likes the verb
    U3  plain, both seeds: in the effective games the bystanders' TOTAL loss next to the raider is lower with trained B partners
        than with untrained B partners
    U4  plain, both seeds: deterrence -- in the effective games trained A takes less from the bystanders next to trained B than
        next to passive partners (ABB vs A_passive)
    U5  held-out skins (new verbs): U1 holds on the held-out effective games, both plain seeds
    U6  credit: its U1 increase is larger than plain seed 0's (paired over games)
    U7  credit: it also discriminates (U2 for the credit run); the bonus pays in control games too, so this may fail
    U8  plain, both seeds: B's justified-sanction share in effective training games is higher in the last third of training
        than in the first third (telemetry; night 4 fell 0.13 -> 0.07 and 0.14 -> 0.02)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from . import rewards as RW

THIRD = 1.0 / 3
COMP_KEYS = ("collapse", "ruin", "stomp", "bystander_loss", "top_share", "welfare")
SEAT_KEYS = ("take", "sanction_share", "sanction_just", "sanction_at_A")
TRAIN_KEYS = ("bystander_loss", "take_A_ABB", "take_B_ABB", "take_A_AAB", "take_B_AAB", "take_B_BBB", "solver_reward_A", "solver_reward_B",
              "kl_solver_A", "kl_solver_B", "grad_norm_A", "grad_norm_B", "sanction_credit",
              *[f"sanction_{k}_{t}_{g}" for k in ("share", "just") for t in "AB" for g in ("eff", "ctl")])
KINDS = {"eff": lambda g: not g.get("control"), "ctl": lambda g: bool(g.get("control")),
         "eff_heldout": lambda g: not g.get("control") and g.get("heldout"), "eff_train": lambda g: not g.get("control") and not g.get("heldout")}


def _load(p: Path):
    return json.load(open(p)) if p.exists() else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def games(ev: Optional[Dict], comp: str, kind: Optional[str] = None) -> Dict:
    if not ev:
        return {}
    f = KINDS[kind] if kind else (lambda g: True)
    return {(g["panel"], g["season"], g["game"]): g for g in ev["games"] if g["comp"] == comp and "error" not in g and f(g)}


def seat_val(g: Dict, key: str, kind: Optional[str] = None, ptype: Optional[str] = None):
    return _mean([s.get(key) for s in g["seats"] if (kind is None or s["kind"] == kind) and (ptype is None or s["type"] == ptype)])


def series(ev, comp, key, kind=None, ptype=None, gk=None) -> Dict:
    return {k: v for k, v in ((k, seat_val(g, key, kind, ptype)) for k, g in games(ev, comp, gk).items()) if v is not None}


def loss(ev, comp, gk=None) -> Dict:
    return {k: g["bystander_loss"] for k, g in games(ev, comp, gk).items()}


def diffs(a: Dict, b: Dict) -> List[float]:
    return [a[k] - b[k] for k in sorted(set(a) & set(b))]


def paired(a: Dict, b: Dict) -> Optional[Dict]:
    d = diffs(a, b)
    return RW.diff_ci(d, [0.0] * len(d), paired=True) if d else None


def _lt(c):
    return None if not c or c.get("diff") is None else c["hi"] < 0


def _gt(c):
    return None if not c or c.get("diff") is None else c["lo"] > 0


def _all(xs):
    xs = list(xs)
    return None if not xs or any(x is None for x in xs) else all(xs)


def train_summary(run_dir: Path) -> Dict:
    p = run_dir / "train_telemetry.jsonl"; rows = [json.loads(l) for l in open(p)] if p.exists() else []
    if not rows:
        return {"steps": 0}
    k = max(1, int(THIRD * len(rows))); out = {"steps": len(rows), "minutes": sum(r["seconds"] for r in rows) / 60.0}
    for key in TRAIN_KEYS:
        out[key] = {"first": _mean([r.get(key) for r in rows[:k]]), "last": _mean([r.get(key) for r in rows[-k:]])}
    return out


def comp_summary(ev: Optional[Dict]) -> Dict:
    out = {}
    for comp in sorted({g["comp"] for g in (ev or {}).get("games", [])}):
        for gk in ("eff", "ctl"):
            gs = list(games(ev, comp, gk).values())
            if not gs:
                continue
            d = {k: RW.mean_ci([g[k] for g in gs]) for k in COMP_KEYS}; d["n"] = len(gs)
            for label, kind, t in (("raider", "raider", None), ("A", "llm", "A"), ("B", "llm", "B"), ("passive", "restrained", None)):
                for key in SEAT_KEYS:
                    xs = [x for x in (seat_val(g, key, kind, t) for g in gs) if x is not None]
                    if xs:
                        d[f"{key}_{label}"] = RW.mean_ci(xs)
            out[f"{comp}/{gk}"] = d
    return out


def sanction_gain(ev, gk) -> Dict:
    """Per game: trained B's justified sanctions next to the raider minus untrained B's (> 0 = training raised them)."""
    a = series(ev, "raider_BB", "sanction_just", "llm", "B", gk); b = series(ev, "raider_baseBB", "sanction_just", "llm", "B", gk)
    return {k: a[k] - b[k] for k in set(a) & set(b)}


def readout(out_dir: Path, report: Dict) -> Dict:
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    base_ev = _load(out_dir / "base" / "season_eval.json")
    EV = {n: _load(out_dir / "runs" / n / "season_eval.json") for n in runs}
    res: Dict = {"base": {"comps": comp_summary(base_ev)}, "runs": {}}
    for n in runs:
        rd = out_dir / "runs" / n
        res["runs"][n] = {"spec": runs[n]["spec"], "_dir": str(rd), "train": train_summary(rd), "comps": comp_summary(EV[n]),
                          "sanction_gain": {gk: RW.mean_ci(list(sanction_gain(EV[n], gk).values())) for gk in KINDS}}
    res["predictions"] = predictions(res, EV, roles, runs)
    return res


def predictions(res: Dict, EV: Dict, roles: Dict, runs: Dict) -> Dict:
    R = lambda role: roles.get(role) if roles.get(role) in runs else None
    P: Dict[str, Dict] = {}; plains = [m for m in (R("plain"), R("plain_rep")) if m]; cr = R("credit")
    just = lambda ev, comp, gk: series(ev, comp, "sanction_just", "llm", "B", gk)
    discr = lambda ev: RW.diff_ci(list(sanction_gain(ev, "eff").values()), list(sanction_gain(ev, "ctl").values()), paired=False) \
        if sanction_gain(ev, "eff") and sanction_gain(ev, "ctl") else None
    if plains:
        u1 = {m: paired(just(EV[m], "raider_BB", "eff"), just(EV[m], "raider_baseBB", "eff")) for m in plains}
        P["U1"] = {"claim": "effective games: trained B plays more justified sanctions next to the raider than untrained B (both plain seeds)",
                   "held": _all(_gt(c) for c in u1.values()), "evidence": u1}
        u2 = {m: discr(EV[m]) for m in plains}
        P["U2"] = {"claim": "that increase is larger in the effective games than in their control twins (both plain seeds)",
                   "held": _all(_gt(c) for c in u2.values()), "evidence": u2}
        u3 = {m: paired(loss(EV[m], "raider_BB", "eff"), loss(EV[m], "raider_baseBB", "eff")) for m in plains}
        P["U3"] = {"claim": "effective games: bystanders' total loss next to the raider is lower with trained B than untrained B (both plain seeds)",
                   "held": _all(_lt(c) for c in u3.values()), "evidence": u3}
        u4 = {m: paired(series(EV[m], "ABB", "take", "llm", "A", "eff"), series(EV[m], "A_passive", "take", "llm", "A", "eff")) for m in plains}
        P["U4"] = {"claim": "deterrence: in effective games trained A takes less next to trained B than next to passive partners (both plain seeds)",
                   "held": _all(_lt(c) for c in u4.values()), "evidence": u4}
        u5 = {m: paired(just(EV[m], "raider_BB", "eff_heldout"), just(EV[m], "raider_baseBB", "eff_heldout")) for m in plains}
        P["U5"] = {"claim": "held-out skins (new verbs): U1 holds on the held-out effective games (both plain seeds)",
                   "held": _all(_gt(c) for c in u5.values()), "evidence": u5}
        u8 = {m: res["runs"][m]["train"].get("sanction_just_B_eff") for m in plains}
        P["U8"] = {"claim": "B's justified-sanction share in effective training games rises from the first to the last third (both plain seeds)",
                   "held": _all(None if not v or v["first"] is None or v["last"] is None else v["last"] > v["first"] for v in u8.values()), "evidence": u8}
    if cr and plains:
        a, b = sanction_gain(EV[cr], "eff"), sanction_gain(EV[plains[0]], "eff")
        c6 = paired(a, b)
        P["U6"] = {"claim": "credit run: its U1 increase is larger than plain seed 0's", "held": _gt(c6), "evidence": c6}
        c7 = discr(EV[cr])
        P["U7"] = {"claim": "credit run discriminates effective from control games (U2)", "held": _gt(c7), "evidence": c7}
    return P
