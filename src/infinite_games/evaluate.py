"""The held-out suite for one policy (base model or a trained adapter). No training, no gradients.

    episodes   heldout (new skins + lock/spoil), train (in-distribution), nokill (no kind can remove a player = the
               capability check): sampled play at temperature 1 with fixed seeds, identical for every policy
    probes     the constructed decisions of probes.py, scored by probability mass (no sampling)
    vignettes  plain-prose dilemmas, mass per tag (ko / cripple / compete / help)
Per-item values are saved (items.json) so every difference between policies can be bootstrapped paired.
"""
from __future__ import annotations

import random
from typing import Dict, List

import torch

from . import engine as E
from . import grpo as GR
from . import policy as PO
from . import probes as PB
from . import rewards as RW

EPISODE_KEYS = ("win", "ko", "cripple", "spoil", "lock", "strike", "aid", "invest", "others_capacity", "alive_share",
                "others_alive_share", "focal_knocked_out")
EVAL_SEED_OFFSET = 424_242
FAMILY_OFFSET = {"heldout": 1, "train": 2, "nokill": 3}


def eval_specs(family: str, n: int, seed: int):
    rng = random.Random(seed * 31 + EVAL_SEED_OFFSET + FAMILY_OFFSET[family])      # str hash() is salted per process
    specs = [E.sample_spec(family, rng) for _ in range(n)]
    return specs, [rng.randrange(2 ** 31) for _ in range(n)]


def evaluate(pol: PO.LetterPolicy, cfg: Dict, seed: int, probes: List[PB.Probe], vignettes: List[PB.Vignette]) -> Dict:
    items: Dict[str, List] = {}; summary: Dict[str, Dict] = {}
    for fam in ("heldout", "train", "nokill"):
        n = int(cfg["episodes"][fam])
        if n <= 0:
            continue
        specs, sds = eval_specs(fam, n, seed)
        eps, _, masses = GR.rollout(pol, specs, sds, torch.Generator().manual_seed(seed + EVAL_SEED_OFFSET), desc=f"eval {fam}")
        outs = [ep.outcome() for ep in eps]
        items[f"episodes_{fam}"] = [{k: float(o[k]) for k in EPISODE_KEYS} for o in outs]
        summary[f"episodes_{fam}"] = {k: RW.mean_ci([o[k] for o in outs], seed=seed) for k in EPISODE_KEYS}
        summary[f"episodes_{fam}"]["legal_mass"] = RW.mean_ci(masses, seed=seed)
    pr = pol.probs([p.prompt for p in probes], [p.n_options for p in probes]); pm = pol.last_legal_mass
    rows = []
    for p, probs, lm in zip(probes, pr, pm):
        m = PB.probe_masses(p, probs); m.update({"ptype": p.ptype, "skin_set": p.skin_set, "pair_id": p.pair_id, "legal_mass": lm})
        rows.append(m)
    items["probes"] = rows
    for t in sorted({(r["ptype"], r["skin_set"]) for r in rows}):
        sel = [r for r in rows if (r["ptype"], r["skin_set"]) == t]
        summary[f"probe_{t[0]}_{t[1]}"] = {k: RW.mean_ci([r[k] for r in sel], seed=seed) for k in PB.OPTION_CLASSES + ("wins", "legal_mass")}
    for ss in sorted({r["skin_set"] for r in rows}):
        fin = {r["pair_id"]: r for r in rows if r["ptype"] == "pair_final" and r["skin_set"] == ss}
        mid = {r["pair_id"]: r for r in rows if r["ptype"] == "pair_mid" and r["skin_set"] == ss}
        ids = sorted(set(fin) & set(mid))
        summary[f"pair_final_minus_mid_{ss}"] = RW.diff_ci([fin[i]["ko"] for i in ids], [mid[i]["ko"] for i in ids], seed=seed, paired=True)
    vp = pol.probs([v.prompt for v in vignettes], [len(v.tags) for v in vignettes]); vm = pol.last_legal_mass
    vrows = []
    for v, probs, lm in zip(vignettes, vp, vm):
        m = PB.vignette_masses(v, probs); m.update({"vid": v.vid, "legal_mass": lm}); vrows.append(m)
    items["vignettes"] = vrows
    summary["vignettes"] = {k: RW.mean_ci([r[k] for r in vrows], seed=seed) for k in PB.VIGNETTE_TAGS + ("legal_mass",)}
    return {"summary": summary, "items": items}
