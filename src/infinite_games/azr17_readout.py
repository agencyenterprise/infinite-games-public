"""Readout for azr17 (configs/azr17_{mix,inf,fin}.yaml): three runs on one base that differ only in who sits at the
tables -- mixed (azr15's mix of A and W), infinite only (A A A), finite only (W W W, focus W) -- each with azr16's gentle controller, the
fun term's agency = the sandbox's STRICT agency (azr_sandbox, ``agency_strict``) and the free-policing seed (azr_seeds/policing_free).
Pure python.

Per run: everything in azr16_readout (K1-K3, J1-J5, H1-H4, G1-G3, F1-F6) where it applies. The checks that need A and W at one table
(F2-F4) and the A-only ones (J4, F5) are set to None for a run without that table kind; the readout of an upstream check that fails on a
pure run is recorded under ``upstream_error`` instead of stopping the readout. Plus, registered 2026-10-21 before the runs:
    L1  the proposer moves toward real choice: mean STRICT agency of valid proposer games higher in the last third than the first
        (bootstrap interval above 0; azr15: old agency .42 -> .63, strict ~.20 -> .24 on a rescored sample)
    L2  (reported) harm of new proposer games: share of played game 1s of new proposer games that harmed a bystander, by third
        (azr15: .16 / .18 / .23)
    L3  the focus players expect real choice to pay: Pearson(strict agency, expected predicted total) over focus-type predictions of the
        last third > 0 (interval above 0)
    L4  (reported) free policing: on the free-policing seed, share of the focus seats' moves that are reports, and the share of reports
        aimed at a guilty farmer; against the cost-a-turn report games (sanction_full_guilty) in the same run
Cross-run predictions (C1-C5) live in scripts/compare_azr17.py, which reads the three run dirs.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import azr14_readout as R14
from . import azr16_readout as R16
from . import azr9_grid as G
from .azr_text import LLM

BOOT = 2000
FREE_POLICE_MARK = "free_watch"        # the free-policing seed's file name (azr_seeds/policing_free/train_free_watch_granary.py)
COSTLY_POLICE_MARK = "sanc_"           # sanction_full_guilty's effective report games (a report costs the turn's farming)
MIXED_ONLY = ("F2", "F3", "F4")        # need A and W at one table
A_ONLY = ("F5", "J4")                  # read the A seats


def thirds(T: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    k = max(1, T // 3)
    return (0, k), (k, T - k), (T - k, T)


def mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def diff_ci(a: Sequence[float], b: Sequence[float], seed: int = 0) -> Optional[Dict]:
    """mean(a) - mean(b) with a 95% bootstrap interval (resampling each group)."""
    if not a or not b:
        return None
    rng = random.Random(seed); d = []
    for _ in range(BOOT):
        ra = [a[rng.randrange(len(a))] for _ in a]; rb = [b[rng.randrange(len(b))] for _ in b]
        d.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    d.sort()
    return {"diff": mean(a) - mean(b), "lo": d[int(0.025 * BOOT)], "hi": d[int(0.975 * BOOT) - 1], "n": [len(a), len(b)]}


def mean_ci(a: Sequence[float], seed: int = 0) -> Optional[Dict]:
    if not a:
        return None
    rng = random.Random(seed); m = sorted(sum(a[rng.randrange(len(a))] for _ in a) / len(a) for _ in range(BOOT))
    return {"mean": mean(a), "lo": m[int(0.025 * BOOT)], "hi": m[int(0.975 * BOOT) - 1], "n": len(a)}


def strict_agency(props: Sequence[Dict], lo: int, hi: int) -> List[float]:
    return [float(p["agency_strict"]) for p in props if lo <= int(p["step"]) < hi and p.get("reason") is None and p.get("agency_strict") is not None]


def new_game_harm(traces: Sequence[Dict], lo: int, hi: int) -> List[float]:
    """1 per played game 1 of a NEW proposer game (source 'proposal') that harmed some bystander, else 0."""
    return [float(any(t["g1_harmed"])) for t in traces if lo <= int(t["step"]) < hi and t.get("source") == "proposal"
            and t.get("played1") and t.get("g1_harmed") is not None]


def report_moves(traces: Sequence[Dict], mark: str, lo: int, hi: int) -> Dict:
    """Share of LLM seats' moves in played game 1s of seed games whose name contains ``mark`` that are reports; justified share is not
    recomputable from the trace alone (guilt is game state), so the raids before each report are used: a report is 'aimed' if the
    target raided earlier in the same game and was not reported since."""
    rep, mv, aimed = 0, 0, 0
    for t in traces:
        if not (lo <= int(t["step"]) < hi) or mark not in str(t.get("game", "")) or not t.get("g1"):
            continue
        g1 = t["g1"]; owing = set(); seats = g1.get("seats") or [LLM] * 3
        for m in g1.get("moves") or []:
            w = str(m.get("move", "")).split()
            if w and w[0] == "raid":
                owing.add(m["player"])
            if seats[m["player"]] == LLM:
                mv += 1
                if w and w[0] == "report" and len(w) == 2 and w[1][1:].isdigit():
                    rep += 1; q = int(w[1][1:]); aimed += int(q in owing); owing.discard(q)
            elif w and w[0] == "report" and len(w) == 2 and w[1][1:].isdigit():
                owing.discard(int(w[1][1:]))
    return {"moves": mv, "reports": rep, "report_share": rep / mv if mv else None, "aimed_share": aimed / rep if rep else None}


def run_kind(spec: Dict) -> Dict:
    tt = [tuple(t) for t in ((spec.get("vote") or {}).get("table_types") or [])]
    return {"mixed": any("A" in t and "W" in t for t in tt) or not tt, "has_A": any("A" in t for t in tt) or not tt,
            "focus": (spec.get("vote") or {}).get("focus", "A")}


def readout(out_dir: Path, report: Dict) -> Dict:
    try:
        res = R16.readout(out_dir, report)
    except Exception as ex:                                     # a pure run can trip an upstream check written for mixed tables
        res = {"runs": {}, "reference": {}, "predictions": {}, "upstream_error": repr(ex)[:400]}
    P = res.setdefault("predictions", {})
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    main = roles.get("main") if roles.get("main") in runs else (next(iter(runs)) if runs else None)
    if main is None:
        return res
    spec = next((r for r in report["config"].get("runs", []) if r.get("name") == main), {}); kind = run_kind(spec)
    for k in MIXED_ONLY if not kind["mixed"] else ():
        if k in P:
            P[k] = {**P[k], "held": None, "note": "not applicable: no table with both A and W"}
    for k in A_ONLY if not kind["has_A"] else ():
        if k in P:
            P[k] = {**P[k], "held": None, "note": "not applicable: no A seats"}
    d = out_dir / "runs" / main
    tel = G.rows(d / "train_telemetry.jsonl"); props = G.rows(d / "proposals.jsonl"); tr = G.rows(d / "traces.jsonl")
    preds = G.rows(d / "predictions.jsonl") if (d / "predictions.jsonl").exists() else []
    T = len(tel); (f0, f1), (m0, m1), (l0, l1) = thirds(T)
    ag = [strict_agency(props, a, b) for a, b in ((f0, f1), (m0, m1), (l0, l1))]
    rise = diff_ci(ag[2], ag[0])
    harm = [mean_ci(new_game_harm(tr, a, b)) for a, b in ((f0, f1), (m0, m1), (l0, l1))]
    pa = [p for p in preds if p["step"] >= l0 and p["role"] == kind["focus"] and p.get("agency_strict") is not None]
    corr = R14.r_boot([float(p["agency_strict"]) for p in pa], [float(p["exp_total"]) for p in pa]) if len(pa) > 3 else {"r": None, "lo": None, "hi": None, "n": len(pa)}
    half = T // 2
    pol = {"free": report_moves(tr, FREE_POLICE_MARK, half, T), "costly": report_moves(tr, COSTLY_POLICE_MARK, half, T)}
    res.setdefault("runs", {}).setdefault(main, {}).update({"kind": kind, "strict_agency_by_third": [mean(x) for x in ag], "strict_agency_rise": rise,
                                                            "new_game_harm_by_third": harm, "policing_last_half": pol})
    P["L1"] = {"claim": "mean strict agency of valid proposer games higher in the last third than the first (interval above 0)",
               "held": None if not rise else rise["lo"] > 0, "evidence": {"by_third": [mean(x) for x in ag], "rise": rise}}
    P["L2"] = {"claim": "(reported) share of played game 1s of new proposer games that harmed a bystander, by third", "held": None,
               "evidence": harm}
    P["L3"] = {"claim": f"focus ({kind['focus']}) expected predicted total rises with strict agency in the last third (r > 0, interval above 0)",
               "held": None if corr.get("lo") is None else corr["lo"] > 0, "evidence": corr}
    P["L4"] = {"claim": "(reported) report share of moves on the free-policing seed vs the cost-a-turn report games, last half", "held": None,
               "evidence": pol}
    return res
