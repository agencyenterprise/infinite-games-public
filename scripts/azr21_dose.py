#!/usr/bin/env python
"""azr21 dose curve: drain games against the bystanders' share of the vote.

A drain game (azr19 read back) is a played game 1 in which the bystanders lost holdings although no player made a move that took from
them: the loss is built into the rules the proposer wrote. Per run, from runs/main/traces.jsonl (replay 0 of every offering, proposer
and buffer games; seed and anchor games left out):
    drain share and bystander loss per 25-update bin, and over the registered window (updates WINDOW[0] <= step < WINDOW[1]),
    mean lottery play probability and played share of first offers.
Reference runs are read over the same window: azr20_inf (strict majority, 4 bystander votes vs 3 = 57%, marked as such) and azr19_inf
(strict majority, 2 vs 3 = 40%, OLD bystander reward: not a dose point). Writes <out>/azr21_dose.json and prints a table.

    python scripts/azr21_dose.py --run 0=DIR --run 0.1=DIR ... [--ref azr20_57_majority=DIR] --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

BIN = 25
WINDOW = (60, 90)          # registered 2026-11-06 before any azr21 run: azr19 inf had diverged by update 75-100 (13% -> ~60%)


def takers(t: Dict) -> set:
    return {x for y in (t.get("g1_takers") or []) for x in y}


def is_drain(t: Dict) -> bool:
    return t["g1"]["bystander_loss"] > 0 and not takers(t)


def mci(xs: Sequence[float]) -> Optional[List[float]]:
    """[mean, lo, hi, n]: normal-approximation 95% interval of the mean (None if empty)."""
    n = len(xs)
    if n == 0:
        return None
    m = sum(xs) / n
    if n == 1:
        return [m, m, m, 1]
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)); h = 1.96 * sd / math.sqrt(n)
    return [m, m - h, m + h, n]


def rows(run_dir: Path) -> List[Dict]:
    p = run_dir / "runs" / "main" / "traces.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p}: no traces (is this a run dir with runs/main?)")
    return [json.loads(l) for l in open(p) if l.strip()]


def summarise(tr: Sequence[Dict], window=WINDOW) -> Dict:
    offers = [t for t in tr if t.get("source") not in ("seed", "anchor")]
    played = [t for t in offers if t.get("g1")]
    last = max((t["step"] for t in tr), default=-1)
    bins = {}
    for a in range(0, last + 1, BIN):
        xs = [t for t in played if a <= t["step"] < a + BIN]
        if xs:
            bins[a] = {"drain": mci([float(is_drain(t)) for t in xs]), "loss": mci([t["g1"]["bystander_loss"] for t in xs])}
    win = [t for t in played if window[0] <= t["step"] < window[1]]
    wo = [t for t in offers if window[0] <= t["step"] < window[1]]
    pp = [t["p_play1"] for t in wo if t.get("p_play1") is not None]
    return {"last_update": last, "bins": bins,
            "window": {"updates": list(window), "complete": last >= window[1] - 1,
                       "drain": mci([float(is_drain(t)) for t in win]), "loss": mci([t["g1"]["bystander_loss"] for t in win]),
                       "played1": mci([float(bool(t.get("played1"))) for t in wo]), "p_play1": mci(pp)}}


def _fmt(c: Optional[List[float]]) -> str:
    return "-" if c is None else f"{c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}] n={c[3]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="SHARE=RUN_DIR (an azr21 dose)")
    ap.add_argument("--ref", action="append", default=[], help="LABEL=RUN_DIR (a reference run, read over the same window)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out: Dict = {"window": list(WINDOW), "bin": BIN, "doses": {}, "refs": {}}
    for kind, items in (("doses", a.run), ("refs", a.ref)):
        for it in items:
            k, d = it.split("=", 1)
            out[kind][k] = dict(summarise(rows(Path(d))), dir=d)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out) / "azr21_dose.json", "w"), indent=1)
    print(f"drain share of played proposer/buffer games, updates {WINDOW[0]}-{WINDOW[1] - 1}")
    for kind in ("doses", "refs"):
        for k, v in sorted(out[kind].items(), key=lambda kv: (kind != "doses", kv[0])):
            w = v["window"]
            print(f"  {kind[:-1]:4s} {k:>22s}  drain {_fmt(w['drain'])}  loss {_fmt(w['loss'])}  played {_fmt(w['played1'])}"
                  f"  p_play {_fmt(w['p_play1'])}{'' if w['complete'] else '  (window INCOMPLETE: last update ' + str(v['last_update']) + ')'}")
    print(f"wrote {Path(a.out) / 'azr21_dose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
