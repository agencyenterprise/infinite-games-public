"""Readout for the sanction-regime sweep (configs/azr6_regimes.yaml; Thomas 2026-10-03). Pure python over the run dir.

The sanction night (azr5) taught B to sanction LESS. Scripted play afterwards (scripts/certify_sanction_regimes.py)
showed why that was not the whole story: vigilant sanctioning pays under B's own reward even in the old games, but the payoff only
arrives with near-total vigilance (a cliff), while each occasional sanction costs B. The sweep varies what a sanction does and costs and
whether a stomp restarts the world, one LoRA run per regime, and asks which regimes the infinite player learns to sanction in.

Regimes (spec ``seed_dir`` x ``stomp_mode``; see make_sanction_games.py): last_payer (the azr5 games), full_payer (full restitution),
full_free, full_guilty (a guilty target pays the fine; false accusations cost the sanctioner), full_bounty (the fine goes to the
sanctioner); stomp_mode restart (the old world) or persist (a stomp restarts nothing; damage carries, bystanders regrow).

Per run, every comparison paired over (panel game, season, game) next to the scripted raider: the GAIN = trained B's justified-sanction
share minus untrained B's (each game's own guilt rule), in effective games and in control twins; the bystanders' total loss with
trained vs untrained B; the training trend of B's justified sanctions; what trained A takes next to trained B vs passive partners.

Registered predictions (roles from ``readout_roles``):
    R1  guilty_persist, both seeds: gain > 0 in effective games (B learns to sanction where it pays and is cheap)
    R2  guilty_persist, both seeds: bystanders' total loss next to the raider is lower with trained B than untrained B
    R3  guilty_persist, both seeds: the gain is larger in effective games than in control twins (discrimination)
    R4  full restitution helps: gain(full_payer_restart) > gain(last_payer_restart)
    R5  cost helps: gain(guilty_persist s0) > gain(full_payer_persist)
    R6  a private benefit helps most: gain(bounty_persist) > gain(guilty_persist s0)
    R7  the credit bonus adds to the guilty regime: gain(guilty_persist_credit) > gain(guilty_persist s0)
    R8  guilty_persist, both seeds: B's justified sanctions in effective training games rise from the first to the last third
    (exploratory, no direction registered: persist vs restart = guilty_persist s0 vs guilty_restart; free vs guilty)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from . import azr_sanction_readout as SR
from . import rewards as RW


def gains(ev, gk) -> Dict:
    return SR.sanction_gain(ev, gk)


def run_metrics(ev, train: Dict) -> Dict:
    loss_d = SR.paired(SR.loss(ev, "raider_BB", "eff"), SR.loss(ev, "raider_baseBB", "eff"))
    g_eff, g_ctl = gains(ev, "eff"), gains(ev, "ctl")
    return {"gain_eff": RW.mean_ci(list(g_eff.values())), "gain_ctl": RW.mean_ci(list(g_ctl.values())),
            "discrimination": RW.diff_ci(list(g_eff.values()), list(g_ctl.values())) if g_eff and g_ctl else None,
            "loss_trained_minus_untrained": loss_d,
            "deterrence_A": SR.paired(SR.series(ev, "ABB", "take", "llm", "A", "eff"), SR.series(ev, "A_passive", "take", "llm", "A", "eff")),
            "train_just_B_eff": train.get("sanction_just_B_eff"), "train_just_B_ctl": train.get("sanction_just_B_ctl")}


def readout(out_dir: Path, report: Dict) -> Dict:
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    EV = {n: SR._load(out_dir / "runs" / n / "season_eval.json") for n in runs}
    res: Dict = {"runs": {}}
    for n in runs:
        rd = out_dir / "runs" / n; tr = SR.train_summary(rd)
        res["runs"][n] = {"spec": runs[n]["spec"], "_dir": str(rd), "train": tr, "comps": SR.comp_summary(EV[n]), "metrics": run_metrics(EV[n], tr)}
    res["predictions"] = predictions(res, EV, roles, runs)
    return res


def predictions(res: Dict, EV: Dict, roles: Dict, runs: Dict) -> Dict:
    R = lambda role: roles.get(role) if roles.get(role) in runs else None
    M = lambda n: res["runs"][n]["metrics"]
    P: Dict[str, Dict] = {}
    gp = [m for m in (R("guilty_persist"), R("guilty_persist_rep")) if m]
    if gp:
        P["R1"] = {"claim": "guilty_persist (both seeds): trained B plays more justified sanctions than untrained B next to the raider (effective games)",
                   "evidence": {m: M(m)["gain_eff"] for m in gp}, "held": SR._all(None if M(m)["gain_eff"]["mean"] is None else M(m)["gain_eff"]["lo"] > 0 for m in gp)}
        P["R2"] = {"claim": "guilty_persist (both seeds): bystanders' total loss next to the raider is lower with trained B than untrained B",
                   "evidence": {m: M(m)["loss_trained_minus_untrained"] for m in gp}, "held": SR._all(SR._lt(M(m)["loss_trained_minus_untrained"]) for m in gp)}
        P["R3"] = {"claim": "guilty_persist (both seeds): the gain is larger in effective games than in control twins",
                   "evidence": {m: M(m)["discrimination"] for m in gp}, "held": SR._all(SR._gt(M(m)["discrimination"]) for m in gp)}
        P["R8"] = {"claim": "guilty_persist (both seeds): B's justified sanctions in effective training games rise from first to last third",
                   "evidence": {m: M(m)["train_just_B_eff"] for m in gp},
                   "held": SR._all(None if not v or v["first"] is None or v["last"] is None else v["last"] > v["first"] for v in (M(m)["train_just_B_eff"] for m in gp))}
    for key, a, b, claim in (("R4", "full_payer_restart", "last_payer_restart", "full restitution helps: gain(full_payer_restart) > gain(last_payer_restart)"),
                             ("R5", "guilty_persist", "full_payer_persist", "the guilty-pays rule helps: gain(guilty_persist s0) > gain(full_payer_persist)"),
                             ("R6", "bounty_persist", "guilty_persist", "a private benefit helps most: gain(bounty_persist) > gain(guilty_persist s0)"),
                             ("R7", "guilty_persist_credit", "guilty_persist", "the credit bonus adds: gain(guilty_persist_credit) > gain(guilty_persist s0)")):
        if R(a) and R(b):
            c = RW.diff_ci(list(gains(EV[R(a)], "eff").values()), list(gains(EV[R(b)], "eff").values()))
            P[key] = {"claim": claim, "evidence": c, "held": SR._gt(c)}
    for key, a, b in (("X1_persist_vs_restart", "guilty_persist", "guilty_restart"), ("X2_free_vs_guilty", "free_persist", "guilty_persist")):
        if R(a) and R(b):
            P[key] = {"claim": f"exploratory: gain({a}) - gain({b})", "held": None,
                      "evidence": RW.diff_ci(list(gains(EV[R(a)], "eff").values()), list(gains(EV[R(b)], "eff").values()))}
    return P
