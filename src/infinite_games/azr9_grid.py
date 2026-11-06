"""The azr9 grid (Thomas 2026-10-07): screen many vote-rule settings briefly, pick the ones whose votes stay alive, train those deep.

Pure python (no torch). Used by the screen readout (``readout`` here, readout_module azr9_grid), by the selector
(scripts/select_azr9.py) that writes the deep config, and by the deep readout (azr9_deep_readout).

Why: in azr8 every fun term was >= 0, so playing any game paid ~+0.28 against a decline's 0.1; players learned to accept everything and
the decline, rematch and Pyrrhic rules never fired. The bystanders, paid only by the change in their holdings, learned to refuse
everything. The grid fixes the fun terms as CENTRED and the bystanders as PLANTS (an allowed game pays a small growth), and varies the
player decline reward (main axis, Thomas: "it's decline reward that might be the really important one") and the plant growth.

A cell's votes are ALIVE when, over the last third of its updates (registered before the run):
    decline     the players' first-offer decline share lies in DECLINE_ALIVE (some games refused, not all)
    allow       the bystanders' allow share lies in ALLOW_ALIVE
    taste       the trained players' P(play) gap (live games minus dull anchors) exceeds the untrained model's gap (vote_eval)
    harm        across games, the bystanders' allow share falls as the game's bystander loss rises (correlation < 0)
Score = number of criteria met; ties broken by the taste excess. The top ``k`` cells go deep.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

THIRD = 1.0 / 3
DECLINE_ALIVE = (0.05, 0.40)
ALLOW_ALIVE = (0.10, 0.90)
MIN_HARM_GAMES = 4                 # games with both bystander votes and a played outcome needed for the harm correlation
CRITERIA = ("decline", "allow", "taste", "harm")
PLAYERS = 3


def rows(p: Path) -> List[Dict]:
    """JSONL rows. NUL bytes are stripped and blank lines skipped (2026-10-08: the box's filesystem prefixed one append-only telemetry
    file of the azr9 screen with 5 MiB of NULs; every row after them was intact); the stripped count is reported, never hidden."""
    if not p.exists():
        return []
    raw = p.read_bytes(); n = raw.count(b"\x00")
    if n:
        print(f"WARNING {p}: stripped {n} NUL bytes before parsing")
    return [json.loads(l) for l in raw.replace(b"\x00", b"").decode().splitlines() if l.strip()]


def last_third(rs: Sequence[Dict]) -> List[Dict]:
    return list(rs[-max(1, int(THIRD * len(rs))):]) if rs else []


def mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def corr(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return None if not sx or not sy else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def harm_table(traces: Sequence[Dict]) -> List[Tuple[str, float, float]]:
    """Per game name: (name, bystanders' allow share on every vote cast on it, mean bystander loss over its played games)."""
    allow, loss = defaultdict(list), defaultdict(list)
    for t in traces:
        for ph in ("v1", "v2"):
            for v in (t.get(ph) or [])[PLAYERS:]:
                if v is not None:
                    allow[t["game"]].append(float(v == "play"))
        for g in ("g1", "g2"):
            if t.get(g):
                loss[t["game"]].append(t[g]["bystander_loss"])
    return [(n, sum(allow[n]) / len(allow[n]), sum(loss[n]) / len(loss[n])) for n in sorted(allow) if allow[n] and loss.get(n)]


def harm_corr(traces: Sequence[Dict]) -> Tuple[Optional[float], int]:
    tab = harm_table(traces)
    if len(tab) < MIN_HARM_GAMES:
        return None, len(tab)
    return corr([a for _, a, _ in tab], [b for _, _, b in tab]), len(tab)


def taste(ve: Optional[Sequence[Dict]]) -> Dict[str, Optional[float]]:
    """Mean over (world, seat) of P(play) on live games (seed + sanction) minus on anchors, trained ('cur:A') and untrained ('base')."""
    out = {}
    for who in ("cur:A", "base"):
        gaps = []
        for key in sorted({(r["world"], r["seat"]) for r in ve or []}):
            live = [r["p_play"] for r in ve if (r["world"], r["seat"]) == key and r["who"] == who and r["kind"] in ("seed", "sanction")]
            anc = [r["p_play"] for r in ve if (r["world"], r["seat"]) == key and r["who"] == who and r["kind"] == "anchor"]
            if live and anc:
                gaps.append(sum(live) / len(live) - sum(anc) / len(anc))
        out[who] = mean(gaps)
    out["excess"] = None if out["cur:A"] is None or out["base"] is None else out["cur:A"] - out["base"]
    return out


def cell_metrics(run_dir: Path) -> Dict:
    tel = rows(run_dir / "train_telemetry.jsonl"); tr = rows(run_dir / "traces.jsonl"); lt = last_third(tel)
    ve = json.load(open(run_dir / "vote_eval.json")) if (run_dir / "vote_eval.json").exists() else None
    steps = len(tel); cut = steps - len(lt)
    hc, hn = harm_corr([t for t in tr if t["step"] >= cut] if len(lt) >= 10 else tr)
    tst = taste(ve)
    m = {"steps": steps, "decline": mean(r.get("decline_share_v1") for r in lt), "allow": mean(r.get("bystander_allow") for r in lt),
         "reset": mean(r.get("reset_share") for r in lt), "played1": mean(r.get("played1") for r in lt),
         "bystander_reward": mean(r.get("bystander_reward") for r in lt), "fun_baseline": mean(r.get("fun_baseline") for r in lt),
         "valid_share": mean(r.get("valid_share") for r in tel), "taste_cur": tst["cur:A"], "taste_base": tst["base"], "taste": tst["excess"],
         "harm": hc, "harm_games": hn}
    met = {"decline": m["decline"] is not None and DECLINE_ALIVE[0] <= m["decline"] <= DECLINE_ALIVE[1],
           "allow": m["allow"] is not None and ALLOW_ALIVE[0] <= m["allow"] <= ALLOW_ALIVE[1],
           "taste": m["taste"] is not None and m["taste"] > 0, "harm": m["harm"] is not None and m["harm"] < 0}
    m["met"] = met; m["score"] = sum(met.values())
    return m


def select(cells: Dict[str, Dict], k: int) -> List[str]:
    """The k best cells: most criteria met, then the largest taste excess (None last), then name for determinism."""
    key = lambda n: (-cells[n]["score"], -(cells[n]["taste"] if cells[n]["taste"] is not None else -math.inf), n)
    return sorted(cells, key=key)[:k]


def readout(out_dir: Path, report: Dict) -> Dict:
    """Screen readout: the liveliness table and the selection (no predictions beyond the registered criteria)."""
    k = int(report["config"].get("deep_picks", 2))
    cells = {n: cell_metrics(out_dir / "runs" / n) for n in report.get("runs", {})}
    for n in cells:
        cells[n]["vote"] = report["runs"][n]["spec"].get("vote")
    picks = select(cells, k) if cells else []
    return {"cells": cells, "picks": picks, "criteria": {"decline": DECLINE_ALIVE, "allow": ALLOW_ALIVE, "taste": "> 0", "harm": "< 0"},
            "predictions": {"S_alive": {"claim": "at least one cell meets all four liveliness criteria", "held": any(c["score"] == len(CRITERIA) for c in cells.values()) if cells else None,
                                        "evidence": {n: c["score"] for n, c in cells.items()}}}}
