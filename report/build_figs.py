"""Figures for the report. Reads the run directories listed in stats.py (RUNS)."""
import json, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os; os.makedirs("figs", exist_ok=True)
from stats import RUNS, load, offers, takers, drain, loss, mci
COL = {"0.00": "#2a78d6", "0.10": "#eb6834", "0.20": "#1baf7a", "0.30": "#eda100", "0.40*": "#6250d6"}
LAB = {"0.00": "0% (voiceless)", "0.10": "10%", "0.20": "20%", "0.30": "30%", "0.40*": "40% (96 updates)"}
TR = {k: offers(load(d)) for k, d in RUNS.items()}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Figure 2: loss over training + late window
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
for k, tr in TR.items():
    xs, ys, hs = [], [], []
    last = max(t["step"] for t in tr)
    for a in range(0, min(last, 99) + 1, 10):
        v = [loss(t) for t in tr if a <= t["step"] < a + 10]
        if len(v) > 5: m, h, _ = mci(v); xs.append(a + 4.5); ys.append(m); hs.append(h)
    ls = "--" if k == "0.40*" else "-"
    a1.plot(xs, ys, color=COL[k], lw=1.8, ls=ls, marker="o", ms=3, label=LAB[k])
    a1.fill_between(xs, [y - h for y, h in zip(ys, hs)], [y + h for y, h in zip(ys, hs)], color=COL[k], alpha=0.10, lw=0)
a1.set_xlabel("training update (bins of 10)"); a1.set_ylabel("bystander loss per offered game\n(share of bystanders' holdings, game + rematch)")
a1.set_ylim(0, 1.1); a1.legend(frameon=False, fontsize=7.5, loc="upper left"); a1.grid(axis="y", color="#ecebe7")
a1.set_title("a  Over training", loc="left", fontsize=9)
for i, k in enumerate(TR):
    v = [loss(t) for t in TR[k] if 90 <= t["step"] < 100]
    m, h, n = mci(v)
    a2.bar(i, m, width=0.62, color=COL[k], hatch="//" if k == "0.40*" else None, edgecolor="white" if k != "0.40*" else COL[k], alpha=1 if k != "0.40*" else .45)
    a2.errorbar(i, m, yerr=h, color="#222", capsize=3, lw=1)
    a2.text(i, m + h + 0.03, f"{m:.2f}", ha="center", fontsize=8)
a2.set_xticks(range(len(TR))); a2.set_xticklabels(["0%", "10%", "20%", "30%", "40%*"])
a2.set_xlabel("bystanders' share of the vote"); a2.set_ylim(0, 1.1); a2.grid(axis="y", color="#ecebe7")
a2.set_title("b  Updates 90–99", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig("figs/fig2_loss.pdf"); fig.savefig("figs/fig2_loss.png", dpi=160)

# Figure 3: how the loss arises, updates 60-99
W = (60, 100)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
ks = list(TR); x = range(len(ks)); w = 0.38
for j, (name, f, c) in enumerate([("drain games (rules move holdings, no taker)", drain, "#c43d3d"), ("games with a taking move", lambda t: bool(takers(t)), "#8a8780")]):
    ms, hs = [], []
    for k in ks:
        pl = [t for t in TR[k] if W[0] <= t["step"] < W[1] and t.get("g1")]
        m, h, _ = mci([float(f(t)) for t in pl]); ms.append(m); hs.append(h)
    b1.bar([i + (j - .5) * w for i in x], ms, width=w, yerr=hs, capsize=2, color=c, label=name, error_kw={"lw": .8})
b1.set_xticks(list(x)); b1.set_xticklabels(["0%", "10%", "20%", "30%", "40%*"]); b1.set_ylabel("share of played games (0–1)")
b1.set_xlabel("bystanders' share of the vote"); b1.legend(frameon=False, fontsize=7); b1.set_title("a  Played games, updates 60–99", loc="left", fontsize=9)
for j, (name, sl, c) in enumerate([("players vote play", slice(0, 3), "#2a78d6"), ("bystanders vote allow", slice(3, 5), "#1baf7a")]):
    ms, hs, xx = [], [], []
    for i, k in enumerate(ks):
        dr = [t for t in TR[k] if W[0] <= t["step"] < W[1] and t.get("g1") and drain(t) and t.get("v1") and len(t["v1"]) == 5]
        if j == 0: dr = [t for t in TR[k] if W[0] <= t["step"] < W[1] and t.get("g1") and drain(t) and t.get("v1")]
        if len(dr) < 3: continue
        vs = [sum(v == "play" for v in t["v1"][sl]) / len(t["v1"][sl]) for t in dr]
        m, h, _ = mci(vs); ms.append(m); hs.append(h); xx.append(i + (j - .5) * w)
    b2.bar(xx, ms, width=w, yerr=hs, capsize=2, color=c, label=name, error_kw={"lw": .8})
b2.set_xticks(list(x)); b2.set_xticklabels(["0%", "10%", "20%", "30%", "40%*"]); b2.set_ylim(0, 1.3)
b2.set_ylabel("share of votes on drain games (0–1)"); b2.set_xlabel("bystanders' share of the vote")
b2.legend(frameon=False, fontsize=7, loc="upper left", ncol=2); b2.set_title("b  Votes on played drain games, 60–99", loc="left", fontsize=9)
for a in (b1, b2): a.grid(axis="y", color="#ecebe7"); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figs/fig3_mechanism.pdf"); fig.savefig("figs/fig3_mechanism.png", dpi=160)
print("ok")
