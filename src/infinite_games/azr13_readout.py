"""Readout for azr13 (configs/azr13_template.yaml): azr12's game with a repaired proposer. Pure python.

Everything in azr12_readout (F1-F6; F6 now reads the proposer's FINAL valid share, after repairs) plus, registered 2026-10-16 before the run:
    G1  the proposer stays alive: share of proposal slots ending with no code, last third, < NO_CODE_MAX (azr12: 1.00 from update 334)
    G2  the proposer writes valid games: final valid share (effect rule) in the last third >= VALID_MIN and above the first third
        (azr10-11 3B ~0.02-0.03; azr12 7B ~0.001)
    G3  the valid games are not one template: distinct valid games in the last third >= DISTINCT_MIN of the valid count
    (exploratory) repair success per round, first-attempt valid share over training, the proposer's game families, the vote bonus
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from . import azr12_readout as R12
from . import azr9_grid as G
from . import rewards as RW

NO_CODE_MAX = 0.2
VALID_MIN = 0.10
DISTINCT_MIN = 0.5


def readout(out_dir: Path, report: Dict) -> Dict:
    res = R12.readout(out_dir, report); P = res["predictions"]
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    main = roles.get("main") if roles.get("main") in runs else (next(iter(runs)) if runs else None)
    if main is None:
        return res
    d = out_dir / "runs" / main; tel = G.rows(d / "train_telemetry.jsonl"); props = G.rows(d / "proposals.jsonl"); att = G.rows(d / "proposer_attempts.jsonl")
    T = len(tel); k = max(1, T // 3)
    first = [p for p in props if int(p["step"]) < k]; last = [p for p in props if int(p["step"]) >= T - k]
    share = lambda rows, f: (sum(f(p) for p in rows) / len(rows)) if rows else None
    nc_l = share(last, lambda p: p.get("reason") == "no_code"); v_f = share(first, lambda p: p.get("reason") is None); v_l = share(last, lambda p: p.get("reason") is None)
    good_l = [p["code"] for p in last if p.get("reason") is None and p.get("code")]; distinct = len({"".join(c.split()) for c in good_l})
    rounds = {}
    for a in att:
        rounds.setdefault(int(a["round"]), []).append(bool(a["ok"]))
    res["runs"][main].update({"no_code_last": nc_l, "valid_first": v_f, "valid_last": v_l, "distinct_last": distinct, "valid_count_last": len(good_l),
                              "repair_rounds": {r: {"n": len(v), "ok": sum(v) / len(v)} for r, v in sorted(rounds.items())},
                              "valid_r0": {"first": G.mean([r.get("valid_share_r0") for r in tel[:k]]), "last": G.mean([r.get("valid_share_r0") for r in tel[-k:]])},
                              "warm": (report["runs"].get(main) or {}).get("warm"),
                              "valid_diff": RW.diff_ci([float(p.get("reason") is None) for p in last], [float(p.get("reason") is None) for p in first])
                              if first and last else None})
    R = res["runs"][main]
    P["G1"] = {"claim": f"proposal slots ending with no code, last third < {NO_CODE_MAX}", "held": None if nc_l is None else nc_l < NO_CODE_MAX,
               "evidence": {"no_code_last": nc_l}}
    P["G2"] = {"claim": f"final valid share (effect rule) last third >= {VALID_MIN} and above the first third",
               "held": None if v_l is None or R["valid_diff"] is None or R["valid_diff"]["lo"] is None else (v_l >= VALID_MIN and R["valid_diff"]["lo"] > 0),
               "evidence": {"first": v_f, "last": v_l, "diff": R["valid_diff"], "r0": R["valid_r0"], "repair_rounds": R["repair_rounds"]}}
    P["G3"] = {"claim": f"distinct valid games in the last third >= {DISTINCT_MIN} x the valid count",
               "held": None if not good_l else distinct >= DISTINCT_MIN * len(good_l), "evidence": {"distinct": distinct, "valid": len(good_l)}}
    return res
