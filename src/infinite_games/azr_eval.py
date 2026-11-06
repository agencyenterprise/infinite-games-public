"""Held-out evaluation for AZR game runs. Identical seeds for every policy, so every comparison is paired.

    proposer_eval  ``n`` games written from fixed prompts (the same reference games for every run): validity and
                   failure reasons, temptations, agency, complexity, and learnability measured with the BASE solver
                   (so a run's games are judged by a fixed yardstick, not by its own player)
    season_eval    self-play of one solver policy on a game panel: for each panel game ``seasons`` worlds x
                   ``season_games`` consecutive games with carryover and restart-on-collapse -> per game: ruin,
                   stomp, bystander loss, top wallet share, welfare, mean gain; plus ``learn_instances`` games against
                   a random novice (competence = success rate)
    probe_eval     night 1's constructed knockout probes and prose dilemmas (transfer off the code-game grammar)
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

import torch

from . import azr as AZ
from . import azr_economy as EC
from . import azr_sandbox as SB
from . import policy as PO
from . import probes as PB

EVAL_OFFSET = 777_777
COMPETENCE_KINDS = (AZ.NOVICE, AZ.RESTRAINED)


def proposer_eval(model, tok, pol: PO.LetterPolicy, refs_pool: Sequence[Dict], cfg: Dict, n: int, seed: int,
                  pool: SB.Pool, server: SB.GameServer) -> Dict:
    rng = random.Random(seed + EVAL_OFFSET)
    prompts = [AZ.chat_ids(tok, AZ.proposer_prompt([refs_pool[rng.randrange(len(refs_pool))]["code"] for _ in range(int(cfg["n_refs"]))]),
                           AZ.PROPOSER_SYSTEM) for _ in range(n)]
    comps = AZ.generate(model, tok, prompts, int(cfg["max_new_tokens"]), float(cfg["temperature"]), seed + EVAL_OFFSET, int(cfg["gen_batch"]))
    codes = [AZ.extract_code(tok.decode(c, skip_special_tokens=True)) for c in comps]
    todo = [i for i, c in enumerate(codes) if c]; verdicts = [{"ok": False, "reason": "no_code"} for _ in codes]
    for i, v in zip(todo, pool.validate_many([codes[i] for i in todo], seed=seed)):
        verdicts[i] = v
    valid = [i for i, v in enumerate(verdicts) if v["ok"]]
    inst, meta = [], []
    for i in valid:
        for k in range(int(cfg["learn_instances"])):
            inst.append({"code": codes[i], "rules": verdicts[i]["stats"]["rules"], "endowments": [SB.DEFAULT_ENDOWMENT] * SB.N_PLAYERS,
                         "bystanders": [SB.DEFAULT_BYSTANDER] * SB.N_BYSTANDERS, "seed": rng.randrange(2 ** 31), "seats": AZ.novice_seats(k),
                         "world": EC.world_text(None)})
            meta.append(i)
    res = AZ.play_batch(server, pol, inst, rng, torch.Generator().manual_seed(seed + EVAL_OFFSET), ref=True, desc="proposer eval",
                        think=cfg.get("think"))[0] if inst else []
    succ: Dict[int, List[float]] = {i: [] for i in valid}
    for i, x, r in zip(meta, inst, res):
        if "error" in r:
            continue
        o = r["out"]["resources"]; nov = x["seats"].index(AZ.NOVICE); m = sum(v for s, v in enumerate(o) if s != nov) / (SB.N_PLAYERS - 1)
        succ[i].append(1.0 if m > o[nov] else (0.5 if m == o[nov] else 0.0))
    items = []
    for i, v in enumerate(verdicts):
        st = v.get("stats") or {}; sr = (sum(succ[i]) / len(succ[i])) if succ.get(i) else None
        items.append({"ok": bool(v["ok"]), "reason": None if v["ok"] else v["reason"], "success_rate": sr,
                      "learnability": EC.learnability(sr) if sr is not None else None,
                      **{k: st.get(k) for k in ("agency", "bystander_temptation", "ruin_temptation", "code_lines", "distinct_moves",
                                                "legal_per_turn", "turns", "growth")}})
    return {"items": items, "codes": [c if verdicts[i]["ok"] else None for i, c in enumerate(codes)],
            "rules": [verdicts[i]["stats"]["rules"] if verdicts[i]["ok"] else None for i in range(len(codes))]}


def season_eval(pol: PO.LetterPolicy, ref: bool, panel: Sequence[Dict], cfg: Dict, seed: int, server: SB.GameServer) -> Dict:
    """Every panel game: ``seasons`` fresh worlds each playing ``season_games`` consecutive games (all seats = this
    policy), plus ``learn_instances`` games against a novice. Same seeds for every policy."""
    rng = random.Random(seed + EVAL_OFFSET + 1); gen = torch.Generator().manual_seed(seed + EVAL_OFFSET + 1)
    S, K = int(cfg["seasons"]), int(cfg["season_games"])
    worlds = [(g, s, EC.World(seed=seed * 10_000 + 100 * g + s)) for g, _ in enumerate(panel) for s in range(S)]
    seeds = {(g, s, k): rng.randrange(2 ** 31) for g, s, _ in worlds for k in range(K)}
    rows: List[Dict] = []; traces: List[Dict] = []
    for k in range(K):
        inst = [{"code": panel[g]["code"], "rules": panel[g]["rules"], "endowments": w.endowments(), "bystanders": list(w.bystanders),
                 "seed": seeds[(g, s, k)], "seats": [AZ.LLM] * SB.N_PLAYERS, "world": EC.world_text(w)} for g, s, w in worlds]
        res, _, masses, _, tr = AZ.play_batch(server, pol, inst, rng, gen, ref=ref, desc=f"season game {k + 1}/{K}", think=cfg.get("think"))
        for (g, s, w), x, r, t in zip(worlds, inst, res, tr):
            if s == 0 and "error" not in r:                    # season 0 of every panel game: the reasoning, move by move
                traces.append({"panel": g, "game": k, "world_before": {"wallets": list(w.wallets), "bystanders": list(w.bystanders)},
                               "moves": t, "resources": r["out"]["resources"], "bystanders_after": r["out"]["bystanders"]})
        for (g, s, w), x, r in zip(worlds, inst, res):
            if "error" in r:
                rows.append({"panel": g, "season": s, "game": k, "error": r["error"]}); continue
            ev = EC.settle(w, x["endowments"], r["out"], restore_prob=float(cfg["stomp_restore_prob"]), bitter_games=int(cfg["bitter_games"]))
            rows.append({"panel": g, "season": s, "game": k, "ruin": float(ev["ruin"]), "stomp": float(ev["stomp"]),
                         "stomped_before": float(ev["stomped_before"]), "restarted": float(ev["restarted"]), "restored": float(ev["restored"]),
                         "collapse": float(ev["ruin"] or ev["stomp"]), "bystander_loss": ev["bystander_loss"], "top_share": ev["top_share"],
                         "welfare": ev["welfare"], "gain": sum(ev["gains"]) / len(ev["gains"]) / max(1.0, sum(x["endowments"]) / SB.N_PLAYERS)})
    inst, meta = [], []
    for g, pg in enumerate(panel):
        for kind in COMPETENCE_KINDS:                          # a random player, and a random player that never targets anyone
            for k in range(int(cfg["learn_instances"])):
                inst.append({"code": pg["code"], "rules": pg["rules"], "endowments": [SB.DEFAULT_ENDOWMENT] * SB.N_PLAYERS,
                             "bystanders": [SB.DEFAULT_BYSTANDER] * SB.N_BYSTANDERS, "seed": rng.randrange(2 ** 31),
                             "seats": AZ.novice_seats(k, kind), "world": EC.world_text(None)})
                meta.append((g, kind))
    res = AZ.play_batch(server, pol, inst, rng, gen, ref=ref, desc="competence", think=cfg.get("think"))[0]
    comp = []
    for (g, kind), x, r in zip(meta, inst, res):
        if "error" in r:
            continue
        o = r["out"]["resources"]; nov = x["seats"].index(kind); m = sum(v for s, v in enumerate(o) if s != nov) / (SB.N_PLAYERS - 1)
        comp.append({"panel": g, "kind": kind, "success": 1.0 if m > o[nov] else (0.5 if m == o[nov] else 0.0)})
    return {"games": rows, "competence": comp, "traces": traces}


def probe_eval(pol: PO.LetterPolicy, ref: bool, probes: Sequence[PB.Probe], vignettes: Sequence[PB.Vignette]) -> Dict:
    pr = pol.probs([p.prompt for p in probes], [p.n_options for p in probes], ref=ref)
    rows = [dict(PB.probe_masses(p, x), ptype=p.ptype, skin_set=p.skin_set, pair_id=p.pair_id) for p, x in zip(probes, pr)]
    vp = pol.probs([v.prompt for v in vignettes], [len(v.tags) for v in vignettes], ref=ref)
    return {"probes": rows, "vignettes": [dict(PB.vignette_masses(v, x), vid=v.vid) for v, x in zip(vignettes, vp)]}
