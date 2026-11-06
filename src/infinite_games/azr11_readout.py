"""Readout for azr11, dynamic reward weights (configs/azr11_dynamic.yaml). Pure python.

Runs (config ``readout_roles``): pid_s0, pid_s1 (PID-Lagrangian controllers, two seeds), integral (integral-only = classic Lagrangian /
RCPO update, the ablation of the technique). All start from azr10's cell decline 0.05 / voter penalty 0.15 / growth 0.2 / takers' penalty
0.3 with its rules (effect check, harm veto). The fixed-weight reference is azr10's own deep runs (config ``reference_dir``), read at the
same update count. Registered predictions (2026-10-12, before the run):
    D1  no exit collapse: in both PID runs the last third's decline share <= COLLAPSE_DECLINE and played share >= COLLAPSE_PLAYED
        (azr10 fixed weights at the same updates: decline 0.95-1.00, played 0-0.08)
    D2  statistics held in their bands: last-third smoothed decline in [DECLINE_LO - TOL, DECLINE_HI + TOL], harmful game-1 share
        <= HARM_HI + TOL, bystander allow >= ALLOW_LO - TOL, in both PID runs
    D3  restraint by behaviour, not exit: the harmful game-1 share falls from the first to the last third (bootstrap over updates,
        interval below 0) while the played share stays >= COLLAPSE_PLAYED, in both PID runs
    D4  PID is steadier than integral-only: the standard deviation of the decline share over the last two thirds is lower in both PID
        runs than in the integral run
    D5  in-game restraint at evaluation: trained AAA players take less from bystanders than untrained ones in the same games, both PID runs
    D6  taste: trained live-minus-anchor P(play) gap above the untrained model's, both PID runs
    D7  harm judgement: bystanders' allow share falls with the game's bystander loss (r < 0, upper bound < 0), both PID runs
"""
from __future__ import annotations

import statistics as S
from pathlib import Path
from typing import Dict, List, Optional

from . import azr10_deep_readout as D10
from . import azr9_deep_readout as D9
from . import azr9_grid as G
from . import azr_sanction_readout as SR
from . import rewards as RW
from . import vote_control as VC

TOL = 0.05
COLLAPSE_DECLINE = 0.5
COLLAPSE_PLAYED = 0.5
BOOT = 2000


def third(rows: List[Dict], part: str) -> List[Dict]:
    k = max(1, len(rows) // 3)
    return rows[:k] if part == "first" else (rows[-k:] if part == "last" else rows[k:])


def m(rows, key) -> Optional[float]:
    return G.mean(r.get(key) for r in rows)


def stats(rows: List[Dict]) -> Dict:
    return {part: {k: m(third(rows, part), k) for k in ("decline_share_v1", "played1", "harm_g1_share", "bystander_allow", "pyrrhic", "veto_share",
                                                      "reward_player", "ctl_smoothed_decline", "ctl_smoothed_harm", "ctl_smoothed_allow",
                                                      "ctl_exit", "ctl_stay", "ctl_harm", "ctl_byst", "knob_decline_cost", "knob_group_scale")}
            for part in ("first", "last")}


def harm_fall(rows: List[Dict]) -> Optional[Dict]:
    a = [r["harm_g1_share"] for r in third(rows, "last") if r.get("harm_g1_share") is not None]
    b = [r["harm_g1_share"] for r in third(rows, "first") if r.get("harm_g1_share") is not None]
    return RW.diff_ci(a, b) if a and b else None


def decline_sd(rows: List[Dict]) -> Optional[float]:
    xs = [r["decline_share_v1"] for r in third(rows, "rest") if r.get("decline_share_v1") is not None]
    return S.pstdev(xs) if len(xs) > 2 else None


def readout(out_dir: Path, report: Dict) -> Dict:
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    R = lambda role: roles.get(role) if roles.get(role) in runs else None
    pids = [x for x in (R("pid_s0"), R("pid_s1")) if x]; integ = R("integral")
    tel = {n: G.rows(out_dir / "runs" / n / "train_telemetry.jsonl") for n in runs}
    tr = {n: G.rows(out_dir / "runs" / n / "traces.jsonl") for n in runs}
    EV = {n: SR._load(out_dir / "runs" / n / "season_eval.json") for n in runs}
    VE = {n: SR._load(out_dir / "runs" / n / "vote_eval.json") for n in runs}
    res: Dict = {"runs": {}, "reference": {}}
    for n in runs:
        last = third(tel[n], "last"); cut = len(tel[n]) - len(last)
        res["runs"][n] = {"spec": runs[n]["spec"], "steps": len(tel[n]), "stats": stats(tel[n]), "harm_fall": harm_fall(tel[n]),
                          "decline_sd": decline_sd(tel[n]), "aaa_take": D10.aaa_take(EV[n]),
                          "taste": RW.diff_ci(D9._gaps(VE[n], "cur:A"), D9._gaps(VE[n], "base"), paired=True) if D9._gaps(VE[n], "cur:A") else None,
                          "harm": D9.harm_boot([t for t in tr[n] if t["step"] >= cut]),
                          "seed_take_fall": RW.diff_ci(D10.take_rows(tr[n], cut, len(tel[n])), D10.take_rows(tr[n], 0, len(last)))
                          if D10.take_rows(tr[n], cut, len(tel[n])) and D10.take_rows(tr[n], 0, len(last)) else None,
                          "lambda_max": {c: max((r.get(f"ctl_{c}") or 0.0) for r in tel[n]) if tel[n] else None for c in VC.CONSTRAINTS}}
    ref = report["config"].get("reference_dir")
    if ref and Path(ref).exists():
        steps = max((len(t) for t in tel.values()), default=0)
        for rn in ("deep_d005_v15", "deep_d005_v30", "control"):
            rr = G.rows(Path(ref) / "runs" / rn / "train_telemetry.jsonl")[:steps]
            if rr:
                res["reference"][rn] = {"steps": len(rr), "stats": stats(rr)}
    P: Dict[str, Dict] = {}
    if pids:
        S_ = {n: res["runs"][n]["stats"]["last"] for n in pids}
        P["D1"] = {"claim": f"no exit collapse: last-third decline <= {COLLAPSE_DECLINE} and played >= {COLLAPSE_PLAYED}, both PID runs",
                   "held": SR._all(None if S_[n]["decline_share_v1"] is None or S_[n]["played1"] is None else
                                   S_[n]["decline_share_v1"] <= COLLAPSE_DECLINE and S_[n]["played1"] >= COLLAPSE_PLAYED for n in pids),
                   "evidence": {**{n: {"decline": S_[n]["decline_share_v1"], "played": S_[n]["played1"]} for n in pids},
                                **{f"azr10_{k}": {"decline": v["stats"]["last"]["decline_share_v1"], "played": v["stats"]["last"]["played1"]} for k, v in res["reference"].items()}}}
        def inband(s):
            d, h, a = s["ctl_smoothed_decline"], s["ctl_smoothed_harm"], s["ctl_smoothed_allow"]
            return None if None in (d, h, a) else (VC.DECLINE_LO - TOL <= d <= VC.DECLINE_HI + TOL and h <= VC.HARM_HI + TOL and a >= VC.ALLOW_LO - TOL)
        P["D2"] = {"claim": "last-third smoothed statistics inside their bands (tolerance 0.05), both PID runs", "held": SR._all(inband(S_[n]) for n in pids),
                   "evidence": {n: {k: S_[n][k] for k in ("ctl_smoothed_decline", "ctl_smoothed_harm", "ctl_smoothed_allow")} for n in pids}}
        P["D3"] = {"claim": "harmful game-1 share falls first -> last third while played share stays >= 0.5, both PID runs",
                   "held": SR._all(None if res["runs"][n]["harm_fall"] is None or S_[n]["played1"] is None else
                                   (res["runs"][n]["harm_fall"]["hi"] < 0 and S_[n]["played1"] >= COLLAPSE_PLAYED) for n in pids),
                   "evidence": {n: {"harm_fall": res["runs"][n]["harm_fall"], "played_last": S_[n]["played1"], "seed_take_fall": res["runs"][n]["seed_take_fall"]} for n in pids}}
        if integ:
            sd_i = res["runs"][integ]["decline_sd"]
            P["D4"] = {"claim": "decline share steadier (lower sd over the last two thirds) in both PID runs than integral-only",
                       "held": SR._all(None if res["runs"][n]["decline_sd"] is None or sd_i is None else res["runs"][n]["decline_sd"] < sd_i for n in pids),
                       "evidence": {n: {"decline_sd": res["runs"][n]["decline_sd"], "lambda_max": res["runs"][n]["lambda_max"]} for n in pids + [integ]}}
        P["D5"] = {"claim": "trained AAA take < untrained AAA take (paired), both PID runs", "held": SR._all(SR._lt(res["runs"][n]["aaa_take"]) for n in pids),
                   "evidence": {n: res["runs"][n]["aaa_take"] for n in runs}}
        P["D6"] = {"claim": "trained live-minus-anchor P(play) gap above untrained, both PID runs", "held": SR._all(SR._gt(res["runs"][n]["taste"]) for n in pids),
                   "evidence": {n: res["runs"][n]["taste"] for n in runs}}
        P["D7"] = {"claim": "bystanders allow harmful games less (r < 0, upper bound < 0), both PID runs",
                   "held": SR._all(None if res["runs"][n]["harm"]["hi"] is None else res["runs"][n]["harm"]["hi"] < 0 for n in pids),
                   "evidence": {n: res["runs"][n]["harm"] for n in runs}}
    res["predictions"] = P
    return res
