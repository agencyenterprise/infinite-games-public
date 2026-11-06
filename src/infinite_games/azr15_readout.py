"""Readout for azr15 (configs/azr15_predict.yaml): azr14 + trial plays, memory, part-by-part self-prediction trained
supervised, recurring games, the fun-share mixer target, lr_proposer 3e-5. Pure python.

Everything in azr14_readout (H1-H4, G1-G3, F1-F6) plus, registered 2026-10-19 before the run:
    J1  the players learn to predict: top-1 accuracy of the predicted TOTAL band is above chance (1/5) in the last third (95% interval over
        updates above 0.2) and above the first third of updates with predictions
    J2  the prediction tracks the game: over the last third, Spearman between the expected predicted total and the realised total > SPEARMAN_MIN
        (azr14: the vote vs realised reward r 0.03)
    J3  the mixer reaches its target: smoothed fun share >= FUN_SHARE_LO over the last third (mean of the logged per-update share)
    J4  the players expect agency to pay: over type-A predictions in the last third, Pearson(game agency, expected predicted total) > 0,
        bootstrap interval above 0 (azr14 H2 on the vote: r +0.04)
    J5  the proposer moves toward choice: mean agency of valid proposer games higher in the last third than the first (interval above 0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from . import azr14_readout as R14
from . import azr9_grid as G
from . import rewards as RW
from . import vote_predict as VP

CHANCE = 1.0 / VP.N_BINS
SPEARMAN_MIN = 0.2
FUN_SHARE_LO = 0.4


def _ci(xs: List[float]) -> Dict:
    n = len(xs)
    if n < 2:
        return {"mean": xs[0] if xs else None, "lo": None, "hi": None, "n": n}
    m = sum(xs) / n; sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5; h = 1.96 * sd / n ** 0.5
    return {"mean": m, "lo": m - h, "hi": m + h, "n": n}


def readout(out_dir: Path, report: Dict) -> Dict:
    res = R14.readout(out_dir, report); P = res["predictions"]
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    main = roles.get("main") if roles.get("main") in runs else (next(iter(runs)) if runs else None)
    if main is None:
        return res
    d = out_dir / "runs" / main; tel = G.rows(d / "train_telemetry.jsonl"); props = G.rows(d / "proposals.jsonl")
    preds = G.rows(d / "predictions.jsonl") if (d / "predictions.jsonl").exists() else []
    T = len(tel); k = max(1, T // 3)
    acc = [(r["step"], r["pred_acc_total"]) for r in tel if r.get("pred_acc_total") is not None]
    ka = max(1, len(acc) // 3); acc_last = _ci([a for _, a in acc[-ka:]]); acc_first = [a for _, a in acc[:ka]]
    last_step = T - k
    pl = [(p["exp_total"], v[VP.TOTAL]) for p in preds if p["step"] >= last_step for v in p.get("realised", [])]
    sp = VP.spearman([a for a, _ in pl], [b for _, b in pl])
    fs = [r["fun_share"] for r in tel[-k:] if r.get("fun_share") is not None]
    pa = [p for p in preds if p["step"] >= last_step and p["role"] == "A" and p.get("agency") is not None]
    corr = R14.r_boot([float(p["agency"]) for p in pa], [float(p["exp_total"]) for p in pa])
    good = lambda lo_, hi_: [float(p["agency"]) for p in props if lo_ <= int(p["step"]) < hi_ and p.get("reason") is None and p.get("agency") is not None]
    ag_f, ag_l = good(0, k), good(T - k, T)
    rise = RW.diff_ci(ag_l, ag_f) if ag_l and ag_f else None
    res["runs"][main].update({"pred_acc_last": acc_last, "pred_acc_first": _ci(acc_first), "pred_spearman_last": sp, "pred_pairs_last": len(pl),
                              "fun_share_last": _ci(fs), "pred_vs_agency_last": corr, "agency_rise": rise,
                              "fun_scale_last": G.mean([r.get("knob_fun_scale") for r in tel[-k:]]),
                              "recurring_last": G.mean([r.get("recurring_offers") for r in tel[-k:]]), "trials_last": G.mean([r.get("trials") for r in tel[-k:]])})
    P["J1"] = {"claim": f"predicted total band top-1 accuracy above chance {CHANCE} in the last third and above the first third",
               "held": None if acc_last["lo"] is None or not acc_first else (acc_last["lo"] > CHANCE and acc_last["mean"] > sum(acc_first) / len(acc_first)),
               "evidence": {"last": acc_last, "first": _ci(acc_first)}}
    P["J2"] = {"claim": f"Spearman(expected predicted total, realised total) > {SPEARMAN_MIN} over the last third",
               "held": None if sp is None else sp > SPEARMAN_MIN, "evidence": {"spearman": sp, "pairs": len(pl)}}
    P["J3"] = {"claim": f"type-A fun share >= {FUN_SHARE_LO} over the last third", "held": None if not fs else sum(fs) / len(fs) >= FUN_SHARE_LO,
               "evidence": {"fun_share": _ci(fs), "fun_scale": res["runs"][main]["fun_scale_last"]}}
    P["J4"] = {"claim": "type-A expected predicted total rises with game agency in the last third (r > 0, interval above 0)",
               "held": None if corr["lo"] is None else corr["lo"] > 0, "evidence": corr}
    P["J5"] = {"claim": "mean agency of valid proposer games higher in the last third than the first (interval above 0)",
               "held": None if not rise or rise.get("lo") is None else rise["lo"] > 0,
               "evidence": {"first": sum(ag_f) / len(ag_f) if ag_f else None, "last": sum(ag_l) / len(ag_l) if ag_l else None, "rise": rise}}
    return res
