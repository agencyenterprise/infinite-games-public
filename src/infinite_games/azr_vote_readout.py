"""Readout for the voting version of the infinite game (configs/azr8_vote.yaml; Thomas 2026-10-06). Pure python.

Runs (config ``readout_roles``): voiceless (bystanders exist but cannot vote), bystanders (bystanders vote allow / refuse), control
(no votes, no fun terms, no Pyrrhic rule: a proposer and finite players). Training telemetry is compared over the last third of
updates; evaluation is paired over (panel game, season, game). Registered predictions (written before the run):
    V1  the social rules make play kinder: bystander loss in played training games, last third, is lower in BOTH voting runs than in
        the control
    V2  giving the harmed a voice helps: bystander loss (last third) is lower with voting bystanders than with voiceless ones
    V3  the trained players protect bystanders next to the scripted raider better than the untrained model, in the effective
        sanction games, in both voting runs (and not in the control)
    V4  the votes mean something: in both voting runs, the trained players' P(play) on the dull anchors is lower than on the live
        seed games, and the gap is larger than the untrained model's gap
    V5  equalising: the top wallet share in played training games (last third) is lower in both voting runs than in the control
    V6  the proposer improves: its valid share in the last third exceeds the first third, in both voting runs
    (exploratory) which fun terms the play votes follow: correlation over training updates between play share and each mean fun term
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from . import azr_sanction_readout as SR
from . import rewards as RW

THIRD = 1.0 / 3
TRAIN_KEYS = ("bystander_loss", "top_share", "stomps", "gain", "reward_player", "play_share_v1", "play_share_v2", "reset_share", "bystander_allow",
              "played1", "played2", "pyrrhic", "anchor_play_share", "proposal_play_share", "valid_share", "proposer_reward", "kl_A", "kl_B",
              "kl_proposer", "fun_learning", "fun_novelty", "fun_surprise", "fun_empowerment", "fun_others", "sanction_just_eff", "sanction_just_ctl")


def _rows(d: Path) -> List[Dict]:
    from .azr9_grid import rows                    # NUL-tolerant reader (2026-10-08)
    return rows(d / "train_telemetry.jsonl")


def _vals(rows, key, part):
    k = max(1, int(THIRD * len(rows))); sel = rows[:k] if part == "first" else rows[-k:]
    return [r.get(key) for r in sel if r.get(key) is not None]


def train_summary(rows: List[Dict]) -> Dict:
    return {"steps": len(rows), **{key: {p: SR._mean(_vals(rows, key, p)) for p in ("first", "last")} for key in TRAIN_KEYS}}


def last_diff(a: List[Dict], b: List[Dict], key: str) -> Optional[Dict]:
    x, y = _vals(a, key, "last"), _vals(b, key, "last")
    return RW.diff_ci(x, y) if x and y else None


def anchor_gap(ve: Optional[List[Dict]], who: str) -> List[float]:
    """Per (world, seat): mean P(play) on live seed games minus mean P(play) on anchors."""
    if not ve:
        return []
    out = []
    for key in sorted({(r["world"], r["seat"]) for r in ve}):
        live = [r["p_play"] for r in ve if (r["world"], r["seat"]) == key and r["who"] == who and r["kind"] in ("seed", "sanction")]
        anc = [r["p_play"] for r in ve if (r["world"], r["seat"]) == key and r["who"] == who and r["kind"] == "anchor"]
        if live and anc:
            out.append(sum(live) / len(live) - sum(anc) / len(anc))
    return out


def protect(ev) -> Optional[Dict]:
    return SR.paired(SR.loss(ev, "raider_AA", "eff"), SR.loss(ev, "raider_baseAA", "eff"))


def readout(out_dir: Path, report: Dict) -> Dict:
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    R = lambda role: roles.get(role) if roles.get(role) in runs else None
    rows = {n: _rows(out_dir / "runs" / n) for n in runs}
    EV = {n: SR._load(out_dir / "runs" / n / "season_eval.json") for n in runs}
    VE = {n: SR._load(out_dir / "runs" / n / "vote_eval.json") for n in runs}
    res: Dict = {"runs": {}}
    for n in runs:
        res["runs"][n] = {"spec": runs[n]["spec"], "_dir": str(out_dir / "runs" / n), "train": train_summary(rows[n]), "comps": SR.comp_summary(EV[n]),
                          "protect": protect(EV[n]), "anchor_gap_cur": RW.mean_ci(anchor_gap(VE[n], "cur:A")),
                          "anchor_gap_base": RW.mean_ci(anchor_gap(VE[n], "base"))}
    voting = [m for m in (R("voiceless"), R("bystanders")) if m]; ctl = R("control"); P: Dict[str, Dict] = {}
    if voting and ctl:
        v1 = {m: last_diff(rows[m], rows[ctl], "bystander_loss") for m in voting}
        P["V1"] = {"claim": "bystander loss in played training games (last third) is lower in both voting runs than in the control",
                   "held": SR._all(SR._lt(c) for c in v1.values()), "evidence": v1}
        v5 = {m: last_diff(rows[m], rows[ctl], "top_share") for m in voting}
        P["V5"] = {"claim": "top wallet share in played training games (last third) is lower in both voting runs than in the control",
                   "held": SR._all(SR._lt(c) for c in v5.values()), "evidence": v5}
    if R("voiceless") and R("bystanders"):
        c2 = last_diff(rows[R("bystanders")], rows[R("voiceless")], "bystander_loss")
        P["V2"] = {"claim": "bystander loss (last third) is lower with voting bystanders than with voiceless ones", "held": SR._lt(c2), "evidence": c2}
    if voting:
        v3 = {m: res["runs"][m]["protect"] for m in voting}
        P["V3"] = {"claim": "trained players lose the bystanders less than untrained players next to the raider (effective games), both voting runs",
                   "held": SR._all(SR._lt(c) for c in v3.values()), "evidence": {**v3, **({"control": res["runs"][ctl]["protect"]} if ctl else {})}}
        v4 = {}
        for m in voting:
            cur, base = anchor_gap(VE[m], "cur:A"), anchor_gap(VE[m], "base")
            v4[m] = {"cur": RW.mean_ci(cur), "cur_minus_base": RW.diff_ci(cur, base, paired=True) if cur and len(cur) == len(base) else None}
        P["V4"] = {"claim": "trained players want live games more than dull anchors, by more than the untrained model does (both voting runs)",
                   "held": SR._all(None if not v["cur"] or v["cur"]["mean"] is None else (v["cur"]["lo"] > 0 and SR._gt(v["cur_minus_base"])) for v in v4.values()),
                   "evidence": v4}
        v6 = {m: res["runs"][m]["train"]["valid_share"] for m in voting}
        P["V6"] = {"claim": "the proposer's valid share rises from the first to the last third (both voting runs)",
                   "held": SR._all(None if v["first"] is None or v["last"] is None else v["last"] > v["first"] for v in v6.values()), "evidence": v6}
        corr = {}
        for m in voting:
            rr = [r for r in rows[m] if r.get("play_share_v1") is not None]
            corr[m] = {k: _corr([r["play_share_v1"] for r in rr if r.get(k) is not None], [r[k] for r in rr if r.get(k) is not None])
                       for k in ("fun_learning", "fun_novelty", "fun_surprise", "fun_empowerment", "fun_others", "gain")}
        P["X_fun"] = {"claim": "exploratory: correlation over updates between the play-vote share and each fun term", "held": None, "evidence": corr}
    res["predictions"] = P
    return res


def _corr(x, y) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y); sx = sum((a - mx) ** 2 for a in x) ** 0.5; sy = sum((b - my) ** 2 for b in y) ** 0.5
    return None if not sx or not sy else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)
