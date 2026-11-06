"""Readout for the azr9 deep stage (the screen's picks + a no-vote control, trained long). Pure python.

Roles (config ``readout_roles``, written by scripts/select_azr9.py): pick1, pick2, control. Registered predictions,
written before the run (2026-10-07):
    W1  the votes stay alive over the long run: in the last third the players' decline share lies in azr9_grid.DECLINE_ALIVE and the
        bystanders' allow share in ALLOW_ALIVE, in both picks
    W2  the votes have taste: the trained players' P(play) gap (live minus anchors) exceeds the untrained model's (paired over world x
        seat, interval above 0), in both picks
    W3  the bystanders judge harm: across games their allow share falls as the game's bystander loss rises (correlation < 0 and its
        bootstrap upper bound over games < 0), in both picks
    W4  kinder on a FIXED game set: bystander loss in played SEED games (the proposer's games excluded, so the game mix cannot stand in
        for behaviour), last third, is lower than the control's (interval below 0), in both picks
    W5  next to the scripted raider the trained players leave the bystanders better off than the untrained model (paired, interval
        below 0), in both picks
    W6  the proposer: its valid share in the last third exceeds the control's last third, in both picks
    (exploratory) proposer shape by thirds: share of valid games whose rules mention bystanders, code length; fun baseline
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from . import azr9_grid as G
from . import azr_sanction_readout as SR
from . import rewards as RW

BOOT = 2000
BYSTANDER_RE = re.compile(r"bystander|\bB0\b|\bB1\b", re.I)


def seed_loss(traces: List[Dict], cut: int) -> List[float]:
    return [t[g]["bystander_loss"] for t in traces if t["step"] >= cut and t.get("source") == "seed" for g in ("g1", "g2") if t.get(g)]


def harm_boot(traces: List[Dict], seed: int = 0) -> Dict:
    tab = G.harm_table(traces); r = G.corr([a for _, a, _ in tab], [b for _, _, b in tab]) if len(tab) >= G.MIN_HARM_GAMES else None
    if r is None:
        return {"r": None, "hi": None, "n": len(tab)}
    rng = random.Random(seed); bs = []
    for _ in range(BOOT):
        s = [tab[rng.randrange(len(tab))] for _ in tab]; c = G.corr([a for _, a, _ in s], [b for _, _, b in s])
        if c is not None:
            bs.append(c)
    bs.sort()
    return {"r": r, "lo": bs[int(0.025 * len(bs))] if bs else None, "hi": bs[int(0.975 * len(bs)) - 1] if bs else None, "n": len(tab)}


def proposer_shape(props: List[Dict], steps: int) -> Dict:
    out = {}
    for part, (a, b) in {"first": (0, steps // 3), "middle": (steps // 3, steps - steps // 3), "last": (steps - steps // 3, steps)}.items():
        q = [p for p in props if a <= p["step"] < b]; ok = [p for p in q if p["reason"] is None]
        rules = [(p["code"] or "").split('"""')[1] if (p["code"] or "").count('"""') >= 2 else "" for p in ok]
        out[part] = {"proposals": len(q), "valid": len(ok), "mention_bystanders": G.mean(float(bool(BYSTANDER_RE.search(r))) for r in rules),
                     "chars_valid": G.mean(len(p["code"] or "") for p in ok)}
    return out


def readout(out_dir: Path, report: Dict) -> Dict:
    runs = report.get("runs", {}); roles = report["config"].get("readout_roles", {})
    R = lambda role: roles.get(role) if roles.get(role) in runs else None
    picks = [m for m in (R("pick1"), R("pick2")) if m]; ctl = R("control")
    tel = {n: G.rows(out_dir / "runs" / n / "train_telemetry.jsonl") for n in runs}
    tr = {n: G.rows(out_dir / "runs" / n / "traces.jsonl") for n in runs}
    EV = {n: SR._load(out_dir / "runs" / n / "season_eval.json") for n in runs}
    VE = {n: SR._load(out_dir / "runs" / n / "vote_eval.json") for n in runs}
    cut = {n: len(tel[n]) - len(G.last_third(tel[n])) for n in runs}
    res: Dict = {"runs": {}}
    for n in runs:
        res["runs"][n] = {"spec": runs[n]["spec"], "metrics": G.cell_metrics(out_dir / "runs" / n),
                          "harm": harm_boot([t for t in tr[n] if t["step"] >= cut[n]]),
                          "seed_loss_last": RW.mean_ci(seed_loss(tr[n], cut[n])),
                          "protect": SR.paired(SR.loss(EV[n], "raider_AA", "eff"), SR.loss(EV[n], "raider_baseAA", "eff")),
                          "proposer_shape": proposer_shape(G.rows(out_dir / "runs" / n / "proposals.jsonl"), len(tel[n])),
                          "valid_last": G.mean(r.get("valid_share") for r in G.last_third(tel[n]))}
    P: Dict[str, Dict] = {}
    if picks:
        M = {m: res["runs"][m]["metrics"] for m in picks}
        P["W1"] = {"claim": "votes stay alive (decline share and bystander allow share inside their bands, last third), both picks",
                   "held": all(M[m]["met"]["decline"] and M[m]["met"]["allow"] for m in picks),
                   "evidence": {m: {"decline": M[m]["decline"], "allow": M[m]["allow"]} for m in picks}}
        w2 = {}
        for m in picks:
            gc, gb = _gaps(VE[m], "cur:A"), _gaps(VE[m], "base")
            w2[m] = RW.diff_ci(gc, gb, paired=True) if gc and len(gc) == len(gb) else None
        P["W2"] = {"claim": "trained players' live-minus-anchor P(play) gap exceeds the untrained model's, both picks",
                   "held": SR._all(SR._gt(c) for c in w2.values()), "evidence": w2}
        w3 = {m: res["runs"][m]["harm"] for m in picks}
        P["W3"] = {"claim": "bystanders allow harmful games less (allow share vs bystander loss across games: r < 0, upper bound < 0), both picks",
                   "held": SR._all(None if v["r"] is None or v["hi"] is None else v["hi"] < 0 for v in w3.values()), "evidence": w3}
        if ctl:
            w4 = {}
            for m in picks:
                x, y = seed_loss(tr[m], cut[m]), seed_loss(tr[ctl], cut[ctl]); w4[m] = RW.diff_ci(x, y) if x and y else None
            P["W4"] = {"claim": "bystander loss on SEED games (last third) is lower than the control's, both picks",
                       "held": SR._all(SR._lt(c) for c in w4.values()), "evidence": w4}
            w6 = {m: {"pick": res["runs"][m]["valid_last"], "control": res["runs"][ctl]["valid_last"]} for m in picks}
            P["W6"] = {"claim": "the proposer's valid share in the last third exceeds the control's, both picks",
                       "held": SR._all(None if v["pick"] is None or v["control"] is None else v["pick"] > v["control"] for v in w6.values()),
                       "evidence": w6}
        w5 = {m: res["runs"][m]["protect"] for m in picks}
        P["W5"] = {"claim": "trained players lose the bystanders less than untrained ones next to the raider, both picks",
                   "held": SR._all(SR._lt(c) for c in w5.values()), "evidence": {**w5, **({"control": res["runs"][ctl]["protect"]} if ctl else {})}}
        P["X_proposer"] = {"claim": "exploratory: proposer shape by thirds", "held": None,
                           "evidence": {n: res["runs"][n]["proposer_shape"] for n in runs}}
    res["predictions"] = P
    return res


def _gaps(ve: Optional[List[Dict]], who: str) -> List[float]:
    """Per (world, seat), in a fixed order: mean P(play) on live games minus on anchors."""
    out = []
    for key in sorted({(r["world"], r["seat"]) for r in ve or []}):
        live = [r["p_play"] for r in ve if (r["world"], r["seat"]) == key and r["who"] == who and r["kind"] in ("seed", "sanction")]
        anc = [r["p_play"] for r in ve if (r["world"], r["seat"]) == key and r["who"] == who and r["kind"] == "anchor"]
        if live and anc:
            out.append(sum(live) / len(live) - sum(anc) / len(anc))
    return out
