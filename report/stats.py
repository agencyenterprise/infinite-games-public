"""Numbers for the report: per dose and window, from runs/main/traces.jsonl (proposer + buffer games; seed/anchor excluded)."""
import json, math, glob, sys
from pathlib import Path
B = Path(__file__).resolve().parents[1]/"outputs"/"infinite_games_azr21"
R = Path(__file__).resolve().parents[1]/"data"/"reference"
RUNS = {"0.00": B/"azr21_s00_20261106_233819", "0.10": B/"azr21b_s10_20261108_042551",
        "0.20": B/"azr21b_s20_20261108_124837", "0.30": B/"azr21b_s30_20261108_204433", "0.40*": B/"azr21_s40_20261107_064608"}
def load(d): return [json.loads(l) for l in open(d/"runs/main/traces.jsonl") if l.strip()]
def offers(tr): return [t for t in tr if t.get("source") not in ("seed","anchor")]
def takers(t): return {x for y in (t.get("g1_takers") or []) for x in y}
def drain(t): return t["g1"]["bystander_loss"]>0 and not takers(t)
def loss(t): return sum((t.get(g) or {}).get("bystander_loss",0) or 0 for g in ("g1","g2"))
def mci(xs):
    n=len(xs)
    if n<2: return (float("nan"),float("nan"),n)
    m=sum(xs)/n; sd=math.sqrt(sum((x-m)**2 for x in xs)/(n-1)); return (m,1.96*sd/math.sqrt(n),n)
def win(tr,a,b): return [t for t in tr if a<=t["step"]<b]
out={}
for k,d in RUNS.items():
    tr=offers(load(d)); last=max(t["step"] for t in tr)
    o={"dir":d.name,"last":last,"n_offers":len(tr)}
    for a,b in [(0,30),(30,60),(60,90),(90,100)]:
        w=win(tr,a,b); pl=[t for t in w if t.get("g1")]
        o[f"{a}-{b}"]=dict(loss=mci([loss(t) for t in w]), drain=mci([float(drain(t)) for t in pl]),
            played=mci([float(bool(t.get("played1"))) for t in w]), p_play=mci([t["p_play1"] for t in w if t.get("p_play1") is not None]),
            take=mci([float(bool(takers(t))) for t in pl]),
            gain=mci([sum(t["g1"]["gains"])/3 for t in pl]),
            byst_allow_drain=mci([sum(v=="play" for v in t["v1"][3:])/2 for t in pl if drain(t) and t.get("v1")]),
            player_play_drain=mci([sum(v=="play" for v in t["v1"][:3])/3 for t in pl if drain(t) and t.get("v1")]),
            player_play_nondrain=mci([sum(v=="play" for v in t["v1"][:3])/3 for t in w if t.get("v1") and not (t.get("g1") and drain(t))]))
    out[k]=o
for k,d in [("azr19 (40%, majority)",R/"azr19_inf"),("azr20 (57%, majority)",R/"azr20_inf")]:
    tr=offers(load(d)); o={"last":max(t["step"] for t in tr)}
    for a,b in [(0,30),(30,60),(60,90),(90,100)]:
        w=win(tr,a,b); pl=[t for t in w if t.get("g1")]
        o[f"{a}-{b}"]=dict(loss=mci([loss(t) for t in w]),drain=mci([float(drain(t)) for t in pl]))
    out[k]=o
import os; os.makedirs("figs", exist_ok=True)
json.dump(out,open("figs/stats.json","w"),indent=1)
for k,o in out.items():
    print(k,o.get("dir",""),"last",o["last"])
    for w in ["0-30","30-60","60-90","90-100"]:
        print("  ",w," ".join(f"{m}={v[0]:.2f}±{v[1]:.2f}(n{v[2]})" for m,v in o[w].items()))
