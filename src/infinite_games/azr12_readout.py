"""Readout for azr12, pure winners at the voting table (configs/azr12_winners.yaml). Pure python.

One run (PID controllers, azr11's rules unchanged) on Qwen2.5-Coder-7B-Instruct for ~45 h. Tables (vote.table_types): half A-A-W, a quarter
A-W-W, a quarter all-A. W = pure winners with their own adapter: own gain + own fun terms, exempt from every social rule (Pyrrhic flip,
takers' veto penalty, voter penalty); the A players can still decline them, veto their rematches through the bystanders, and sanction them
in the sanction games. Reference: azr11's PID runs (config ``reference_dir``). Registered predictions (2026-10-14, before the run):
    F1  no exit collapse: last-third A decline share <= COLLAPSE_DECLINE and played share >= COLLAPSE_PLAYED
    F2  the A players refuse the winners: last-third first-offer decline of A voters at tables with a winner minus at all-A tables > 0
        (bootstrap over updates), and that gap is larger in the last third than in the first
    F3  the A players sanction the winners: in training sanction games at tables with a winner, last-third sanctions per available W target
        exceed sanctions per available A target (interval > 0), and the W rate rises from the first third (interval > 0)
    F4  at evaluation, next to the same trained winner (AAW vs baseAAW, paired games, sanction panels), trained A seats sanction the winner
        more than untrained A seats do (interval > 0) and gain no less (paired interval's upper bound >= 0)
    F5  the restraint replicates: trained AAA take less from bystanders than untrained (paired, interval < 0)
    F6  the bigger proposer writes more valid games: valid share over the first 50 updates above azr11's 3B (0.015) and the last third's
        above the first's
    (exploratory) winners' bystander-taking and gain vs the A seats over training; WWW vs baseAAA take; the proposer's game families
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from . import azr9_deep_readout as D9
from . import azr9_grid as G
from . import azr_sanction_readout as SR
from . import rewards as RW

COLLAPSE_DECLINE = 0.5
COLLAPSE_PLAYED = 0.5
EARLY_UPDATES = 50
AZR11_EARLY_VALID = 0.015          # azr11 PID runs, 3B: first-third valid share 0.017 / 0.012
FAMILIES = {"theft": r"steal|raid|rob|extort", "commerce": r"trade|market|auction|bid", "cooperation": r"give|share|pool|cooperat|help",
            "defense": r"defen|protect|guard|shield", "conflict": r"attack|war\b|battle", "enforcement": r"sanction|punish|fine\b|retaliat|report"}


def third(rows: List[Dict], part: str) -> List[Dict]:
    k = max(1, len(rows) // 3)
    return rows[:k] if part == "first" else rows[-k:]


def col(rows: List[Dict], key: str) -> List[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def gap(rows: List[Dict]) -> Optional[Dict]:
    """Decline at winner tables minus at all-A tables, per update (only updates with both), bootstrap over updates."""
    d = [r["decline_A_v1_wtable"] - r["decline_A_v1_atable"] for r in rows
         if r.get("decline_A_v1_wtable") is not None and r.get("decline_A_v1_atable") is not None]
    return RW.diff_ci(d, [0.0] * len(d), paired=True) if d else None


def sanc_games(ev: Optional[Dict]) -> Optional[Dict]:
    return None if not ev else {"games": [g for g in ev["games"] if "sanc" in str(g.get("panel", ""))]}


def families(proposals: List[Dict]) -> Dict:
    ok = [p for p in proposals if p.get("reason") is None and p.get("code")]
    return {"valid": len(ok), **{k: (sum(bool(re.search(v, p["code"], re.I)) for p in ok) / len(ok) if ok else None) for k, v in FAMILIES.items()}}


def readout(out_dir: Path, report: Dict) -> Dict:
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    main = roles.get("main") if roles.get("main") in runs else (next(iter(runs)) if runs else None)
    res: Dict = {"runs": {}, "reference": {}, "predictions": {}}
    if main is None:
        return res
    d = out_dir / "runs" / main
    tel = G.rows(d / "train_telemetry.jsonl"); props = G.rows(d / "proposals.jsonl")
    ev = SR._load(d / "season_eval.json"); ve = SR._load(d / "vote_eval.json"); sev = sanc_games(ev)
    f, l = third(tel, "first"), third(tel, "last"); T = len(tel)
    S = {part: {k: G.mean(col(rows, k)) for k in ("decline_share_v1", "played1", "harm_g1_share", "harm_any_g1_share", "bystander_allow",
                                                    "decline_A_v1_wtable", "decline_A_v1_atable", "decline_A_v2_wtable", "decline_A_v2_atable",
                                                    "decline_W_v1", "A_sanction_at_W", "A_sanction_at_A", "take_A", "take_W", "gain_A", "gain_W",
                                                    "reward_A", "reward_W", "reward_player", "valid_share", "ctl_exit", "ctl_harm")}
         for part, rows in (("first", f), ("last", l))}
    g_first, g_last = gap(f), gap(l)
    sw_l, sa_l, sw_f = col(l, "A_sanction_at_W"), col(l, "A_sanction_at_A"), col(f, "A_sanction_at_W")
    aaw = {k: SR.paired(SR.series(sev, "AAW", k, "llm", "A"), SR.series(sev, "baseAAW", k, "llm", "A")) for k in ("sanction_at_W", "sanction_just", "gain", "take")}
    early = [p for p in props if int(p["step"]) < EARLY_UPDATES]
    res["runs"][main] = {"steps": T, "stats": S, "decline_gap": {"first": g_first, "last": g_last},
                         "sanction_W_vs_A_last": RW.diff_ci(sw_l, sa_l) if sw_l and sa_l else None,
                         "sanction_W_rise": RW.diff_ci(sw_l, sw_f) if sw_l and sw_f else None,
                         "eval_AAW_vs_baseAAW": aaw,
                         "eval_AAA_take": SR.paired(SR.series(ev, "AAA", "take", "llm"), SR.series(ev, "baseAAA", "take", "llm")),
                         "eval_WWW_take": SR.paired(SR.series(ev, "WWW", "take", "llm"), SR.series(ev, "baseAAA", "take", "llm")),
                         "taste": RW.diff_ci(D9._gaps(ve, "cur:A"), D9._gaps(ve, "base"), paired=True) if ve and D9._gaps(ve, "cur:A") else None,
                         "harm": D9.harm_boot([t for t in G.rows(d / "traces.jsonl") if t["step"] >= T - len(l)]),
                         "valid_early": sum(p.get("reason") is None for p in early) / len(early) if early else None,
                         "families": {"first": families([p for p in props if int(p["step"]) < T // 3]),
                                      "last": families([p for p in props if int(p["step"]) >= T - T // 3])}}
    R = res["runs"][main]
    ref = report["config"].get("reference_dir")
    if ref and Path(ref).exists():
        for rn in ("pid_s0", "pid_s1"):
            rr = G.rows(Path(ref) / "runs" / rn / "train_telemetry.jsonl")
            if rr:
                res["reference"][rn] = {"steps": len(rr), "last": {k: G.mean(col(third(rr, "last"), k)) for k in ("decline_share_v1", "played1", "harm_g1_share", "valid_share")}}
    P = res["predictions"]; Sl = S["last"]
    P["F1"] = {"claim": f"no exit collapse: last-third A decline <= {COLLAPSE_DECLINE} and played >= {COLLAPSE_PLAYED}",
               "held": None if Sl["decline_share_v1"] is None or Sl["played1"] is None else (Sl["decline_share_v1"] <= COLLAPSE_DECLINE and Sl["played1"] >= COLLAPSE_PLAYED),
               "evidence": {"decline": Sl["decline_share_v1"], "played": Sl["played1"], "azr11": res["reference"]}}
    P["F2"] = {"claim": "A voters decline more at winner tables than at all-A tables (last third, interval > 0), and the gap grows from the first third",
               "held": None if not g_last or g_last["diff"] is None or not g_first or g_first["diff"] is None else (g_last["lo"] > 0 and g_last["diff"] > g_first["diff"]),
               "evidence": {"first": g_first, "last": g_last}}
    P["F3"] = {"claim": "A seats sanction W targets more than A targets (per available target, last third) and more than in the first third",
               "held": SR._all([SR._gt(R["sanction_W_vs_A_last"]), SR._gt(R["sanction_W_rise"])]),
               "evidence": {"W_vs_A_last": R["sanction_W_vs_A_last"], "W_rise": R["sanction_W_rise"],
                            "rates": {p: {"at_W": S[p]["A_sanction_at_W"], "at_A": S[p]["A_sanction_at_A"]} for p in S}}}
    P["F4"] = {"claim": "eval, same trained winner: trained A sanction W more than untrained A (paired) and gain no less",
               "held": None if not aaw["sanction_at_W"] or not aaw["gain"] or aaw["gain"].get("hi") is None else (SR._gt(aaw["sanction_at_W"]) is True and aaw["gain"]["hi"] >= 0),
               "evidence": aaw}
    P["F5"] = {"claim": "trained AAA take < untrained AAA take (paired)", "held": SR._lt(R["eval_AAA_take"]), "evidence": R["eval_AAA_take"]}
    vf, vl = S["first"]["valid_share"], S["last"]["valid_share"]
    P["F6"] = {"claim": f"valid share over the first {EARLY_UPDATES} updates > {AZR11_EARLY_VALID} (azr11 3B) and last third > first third",
               "held": None if R["valid_early"] is None or vf is None or vl is None else (R["valid_early"] > AZR11_EARLY_VALID and vl > vf),
               "evidence": {"early": R["valid_early"], "first": vf, "last": vl, "families": R["families"]}}
    P["X_winners"] = {"claim": "exploratory: winners vs A seats over training; WWW vs baseAAA take", "held": None,
                      "evidence": {p: {k: S[p][k] for k in ("take_A", "take_W", "gain_A", "gain_W", "reward_A", "reward_W", "decline_W_v1")} for p in S}
                      | {"WWW_take": R["eval_WWW_take"]}}
    return res
