"""Readout for azr16 (configs/azr16_gentle.yaml): azr15 with a gentler controller (derivative gain 0.5, smoothing 0.1, every
multiplier moves at most max_step per update). Pure python.

Everything in azr15_readout (J1-J5, H1-H4, G1-G3, F1-F6) plus, registered 2026-10-20 before the run:
    K1  no runaway decline: after update STABLE_FROM, no WINDOW-update window has a mean type-A decline share above DECLINE_HIGH or below
        DECLINE_LOW (azr15: 0.85 around update 95-110, then ~0 for 45 updates)
    K2  the step limit holds: no controller multiplier changes by more than max_step between consecutive updates (mechanical check)
    K3  the vote keeps choosing: correlation between voting play and the predicted total stays above VOTE_PRED_MIN in the last third
        (azr15: 0.60 in the middle, 0.20 under always-play)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from . import azr15_readout as R15
from . import azr9_grid as G

STABLE_FROM = 50
WINDOW = 20
DECLINE_HIGH = 0.6
DECLINE_LOW = 0.02
VOTE_PRED_MIN = 0.3
STEP_EPS = 1e-6


def readout(out_dir: Path, report: Dict) -> Dict:
    res = R15.readout(out_dir, report); P = res["predictions"]
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    main = roles.get("main") if roles.get("main") in runs else (next(iter(runs)) if runs else None)
    if main is None:
        return res
    d = out_dir / "runs" / main; tel = G.rows(d / "train_telemetry.jsonl"); T = len(tel); k = max(1, T // 3)
    dec = [r.get("decline_share_v1") for r in tel]
    wins = []
    for lo in range(STABLE_FROM, max(STABLE_FROM, T - WINDOW + 1)):
        w = [x for x in dec[lo:lo + WINDOW] if x is not None]
        if w:
            wins.append((lo, sum(w) / len(w)))
    worst_hi = max(wins, key=lambda t: t[1]) if wins else None; worst_lo = min(wins, key=lambda t: t[1]) if wins else None
    ms = (report["config"].get("runs", [{}])[0].get("vote", {}).get("control") or {}).get("max_step")
    jumps = [abs((b.get(c) or 0) - (a.get(c) or 0)) for a, b in zip(tel, tel[1:]) for c in [x for x in b if x.startswith("ctl_") and not x.startswith("ctl_smoothed")]]
    max_jump = max(jumps) if jumps else None
    vote = {}
    for t in G.rows(d / "traces.jsonl"):
        for s, v in enumerate((t.get("v1") or [])[:3]):
            vote[(t["step"], t["j"], s)] = float(v in ("play", "reset"))
    preds = G.rows(d / "predictions.jsonl") if (d / "predictions.jsonl").exists() else []
    pv = [(p["exp_total"], vote[(p["step"], p["j"], p["seat"])]) for p in preds if p["step"] >= T - k and (p["step"], p["j"], p["seat"]) in vote]
    r = float(np.corrcoef(*zip(*pv))[0, 1]) if len(pv) > 3 and np.std([a for a, _ in pv]) > 0 and np.std([b for _, b in pv]) > 0 else None
    res["runs"][main].update({"decline_window_max": worst_hi, "decline_window_min": worst_lo, "ctl_max_jump": max_jump, "vote_pred_last": r})
    P["K1"] = {"claim": f"after update {STABLE_FROM}, no {WINDOW}-update window with mean A decline > {DECLINE_HIGH} or < {DECLINE_LOW}",
               "held": None if not wins else (worst_hi[1] <= DECLINE_HIGH and worst_lo[1] >= DECLINE_LOW), "evidence": {"max": worst_hi, "min": worst_lo}}
    P["K2"] = {"claim": f"no controller multiplier moves more than max_step ({ms}) per update",
               "held": None if ms is None or max_jump is None else max_jump <= float(ms) + STEP_EPS, "evidence": {"max_jump": max_jump}}
    P["K3"] = {"claim": f"r(play vote, predicted total) > {VOTE_PRED_MIN} in the last third", "held": None if r is None else r > VOTE_PRED_MIN,
               "evidence": {"r": r, "n": len(pv)}}
    return res
