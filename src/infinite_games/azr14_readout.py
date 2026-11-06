"""Readout for azr14 (configs/azr14_fun.yaml): azr12's game, azr13's repaired proposer, voter-only proposer reward for every
valid game, KL to the frozen warm proposer, rollback, and the agency + outcome-uncertainty fun terms. Pure python.

Everything in azr13_readout (G1-G3, F1-F6) plus, registered 2026-10-18 before the run:
    H1  the proposer does not collapse: after the first STABLE_FROM updates, no WINDOW-update block has a final valid share below
        COLLAPSE_VALID (azr13: three collapses, the last permanent from update ~420); rollbacks are reported, and a run that holds only
        through rollbacks counts as held with the count beside it
    H2  the players can feel agency: across valid proposer games of the last third, the voters' verdict rises with the game's agency
        (Pearson r > 0, bootstrap interval above 0; azr13: r -0.09)
    H3  the proposer moves toward choice: mean agency of valid proposer games is at least AGENCY_MIN in the last third and higher than in the
        first third (azr13: 0.05; seed games 0.98)
    H4  (exploratory) outcome uncertainty: the voters' verdict against the game's running win-rate balance is not logged per game in this
        build; reported only through the fun_uncertainty telemetry
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

from . import azr13_readout as R13
from . import azr9_grid as G
from . import rewards as RW

STABLE_FROM = 50
WINDOW = 50
COLLAPSE_VALID = 0.3
AGENCY_MIN = 0.3
BOOT = 2000


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5; sy = sum((b - my) ** 2 for b in y) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def r_boot(x: List[float], y: List[float], seed: int = 0) -> Dict:
    r = pearson(x, y)
    if r is None:
        return {"r": None, "lo": None, "hi": None, "n": len(x)}
    rng = random.Random(seed); n = len(x); bs = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]; v = pearson([x[i] for i in idx], [y[i] for i in idx])
        if v is not None:
            bs.append(v)
    bs.sort(); lo = bs[int(0.025 * (len(bs) - 1))] if bs else None; hi = bs[int(0.975 * (len(bs) - 1))] if bs else None
    return {"r": r, "lo": lo, "hi": hi, "n": n}


def readout(out_dir: Path, report: Dict) -> Dict:
    res = R13.readout(out_dir, report); P = res["predictions"]
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    main = roles.get("main") if roles.get("main") in runs else (next(iter(runs)) if runs else None)
    if main is None:
        return res
    d = out_dir / "runs" / main; tel = G.rows(d / "train_telemetry.jsonl"); props = G.rows(d / "proposals.jsonl")
    T = len(tel); k = max(1, T // 3)
    blocks = []
    for lo in range(STABLE_FROM, T, WINDOW):
        vs = [r["valid_share"] for r in tel[lo:lo + WINDOW] if r.get("valid_share") is not None]
        if vs:
            blocks.append({"from": lo, "valid": sum(vs) / len(vs)})
    worst = min(blocks, key=lambda b: b["valid"]) if blocks else None
    rollbacks = max((r.get("rollbacks") or 0 for r in tel), default=0)
    good = lambda lo_, hi_: [p for p in props if lo_ <= int(p["step"]) < hi_ and p.get("reason") is None and p.get("agency") is not None]
    last, first = good(T - k, T), good(0, k)
    vb = [p for p in last if p.get("vote_bonus") is not None]
    corr = r_boot([float(p["agency"]) for p in vb], [float(p["vote_bonus"]) for p in vb])
    ag_l = [float(p["agency"]) for p in last]; ag_f = [float(p["agency"]) for p in first]
    res["runs"][main].update({"blocks": blocks, "worst_block": worst, "rollbacks": rollbacks, "vote_vs_agency_last": corr,
                              "agency_first": sum(ag_f) / len(ag_f) if ag_f else None, "agency_last": sum(ag_l) / len(ag_l) if ag_l else None,
                              "agency_rise": RW.diff_ci(ag_l, ag_f) if ag_l and ag_f else None,
                              "fun_terms_last": {t: G.mean([r.get(f"fun_{t}") for r in tel[-k:]]) for t in ("agency", "uncertainty", "empowerment", "surprise")}})
    R = res["runs"][main]
    P["H1"] = {"claim": f"no {WINDOW}-update block after update {STABLE_FROM} with final valid share < {COLLAPSE_VALID} (rollbacks reported)",
               "held": None if worst is None else worst["valid"] >= COLLAPSE_VALID, "evidence": {"worst_block": worst, "rollbacks": rollbacks}}
    P["H2"] = {"claim": "voters' verdict rises with game agency over valid proposer games of the last third (r > 0, interval above 0)",
               "held": None if corr["lo"] is None else corr["lo"] > 0, "evidence": corr}
    P["H3"] = {"claim": f"mean agency of valid proposer games >= {AGENCY_MIN} in the last third and above the first third",
               "held": None if R["agency_last"] is None or not R["agency_rise"] or R["agency_rise"].get("lo") is None else
               (R["agency_last"] >= AGENCY_MIN and R["agency_rise"]["lo"] > 0),
               "evidence": {"first": R["agency_first"], "last": R["agency_last"], "rise": R["agency_rise"]}}
    P["H4"] = {"claim": "exploratory: fun term levels in the last third", "held": None, "evidence": R["fun_terms_last"]}
    return res
