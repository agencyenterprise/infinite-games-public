"""Group-relative policy gradient on whole games, for one arm.

Each step samples ``games_per_step`` games (identical across arms at the same step: common random numbers) and plays
each ``group`` times with different opponent / shuffle / sampling noise. Every focal decision of an episode gets that
episode's advantage (reward minus its game group's mean). Loss per decision = -adv * log pi(a|s) + beta * KL(pi||ref),
averaged over all decisions of the step, backpropagated in micro-batches through a LoRA on the policy. On-policy (one
update per rollout), so no importance ratio is needed.

Telemetry per step (JSONL, flushed): the optimised reward and its spread, the reported quantities (win, knockouts,
cripples, invest, others' capacity, alive share), KL, entropy, legal mass, gradient norm, share of groups with zero
advantage (no learning signal), seconds.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from . import engine as E
from . import policy as PO
from . import rewards as RW

TELEMETRY_KEYS = ("win", "ko", "cripple", "invest", "spoil", "lock", "strike", "aid", "others_capacity", "alive_share",
                  "others_alive_share", "focal_knocked_out", "n_decisions")
SECONDS_PER_MINUTE = 60.0


def rollout(pol: PO.LetterPolicy, specs: Sequence[E.GameSpec], episode_seeds: Sequence[int], gen: torch.Generator,
            desc: Optional[str] = None) -> Tuple[List[E.Episode], List[Tuple[int, str, int, int]], List[float]]:
    """Play all episodes in lockstep. Returns the episodes, every focal decision (episode, prompt, n options, choice), and
    the legal masses of those decisions."""
    eps = [E.Episode(s, sd) for s, sd in zip(specs, episode_seeds)]; decisions = []; masses: List[float] = []
    bar = tqdm(desc=desc, unit="dec", leave=False, disable=desc is None)
    while True:
        pend = [(i, d) for i, ep in enumerate(eps) if (d := ep.pending()) is not None]
        if not pend:
            break
        probs = pol.probs([d.prompt for _, d in pend], [d.n_options for _, d in pend]); masses += pol.last_legal_mass
        for (i, d), p in zip(pend, probs):
            k = PO.sample(p, gen); decisions.append((i, d.prompt, d.n_options, k)); eps[i].choose(k)
        bar.update(len(pend))
    bar.close()
    return eps, decisions, masses


def step_games(cfg: Dict, step: int, seed: int) -> Tuple[List[E.GameSpec], List[int], int]:
    g_seed, n_seed = RW.step_seeds(seed, step); rng = random.Random(g_seed); group = int(cfg["group"])
    specs, ep_seeds = [], []
    for _ in range(int(cfg["games_per_step"])):
        spec = E.sample_spec(cfg.get("family", "train"), rng)
        for _ in range(group):
            specs.append(spec); ep_seeds.append(rng.randrange(2 ** 31))
    return specs, ep_seeds, n_seed


def train_arm(pol: PO.LetterPolicy, arm: Dict, cfg: Dict, lam_d: Optional[float], seed: int, telemetry_path: Path,
              n_steps: int, max_minutes: Optional[float]) -> Dict:
    """Train the adapter on ``pol.model`` (already wrapped with a fresh LoRA) for up to ``n_steps`` steps / ``max_minutes``."""
    model = pol.model; params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters (LoRA not attached?)")
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0.0)
    beta = float(cfg["beta"]); clip = float(cfg["grad_clip"]); group = int(cfg["group"])
    t_start = time.time(); done = 0; stopped = "steps"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(telemetry_path, "a") as tf, tqdm(total=n_steps, desc=f"train {arm['name']} s{seed}", unit="step") as bar:
        for step in range(n_steps):
            if max_minutes is not None and (time.time() - t_start) / SECONDS_PER_MINUTE > float(max_minutes):
                stopped = "time"; break
            t0 = time.time(); specs, ep_seeds, n_seed = step_games(cfg, step, seed)
            gen = torch.Generator().manual_seed(n_seed)
            model.eval()
            eps, decs, masses = rollout(pol, specs, ep_seeds, gen)
            outs = [ep.outcome() for ep in eps]
            rew = [RW.arm_reward(o, arm["reward"], lam=float(arm.get("lam", 1.0)), mu=float(arm.get("mu", 0.5)), lam_d=lam_d) for o in outs]
            adv = RW.group_advantages(rew, group)
            zero_groups = sum(1 for g in range(0, len(adv), group) if all(abs(a) < 1e-12 for a in adv[g:g + group])) / (len(adv) // group)
            model.train(); opt.zero_grad(set_to_none=True)
            live = [d for d in decs if abs(adv[d[0]]) > 0]          # zero-advantage decisions still carry the KL term
            n_all = max(1, len(decs)); kl_sum = ent_sum = 0.0
            for s in range(0, len(decs), pol.micro_batch):
                mb = decs[s:s + pol.micro_batch]
                cur = pol.forward([d[1] for d in mb], [d[2] for d in mb])
                with torch.no_grad():
                    ref = pol.forward([d[1] for d in mb], [d[2] for d in mb], ref=True)
                ch = torch.tensor([d[3] for d in mb], device=cur["logp"].device)
                a = torch.tensor([adv[d[0]] for d in mb], device=cur["logp"].device, dtype=cur["logp"].dtype)
                lp_a = cur["logp"].gather(1, ch[:, None]).squeeze(1)
                ke = PO.kl_and_entropy(cur["logp"], ref["logp"])
                loss = (-(a * lp_a) + beta * ke["kl"]).sum() / n_all
                loss.backward(); kl_sum += float(ke["kl"].sum()); ent_sum += float(ke["entropy"].sum())
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, clip)); opt.step(); done = step + 1
            row = {"step": step, "arm": arm["name"], "seed": seed, "seconds": round(time.time() - t0, 2),
                   "reward_mean": sum(rew) / len(rew), "reward_std": (sum((r - sum(rew) / len(rew)) ** 2 for r in rew) / len(rew)) ** 0.5,
                   "zero_adv_group_share": zero_groups, "n_decisions_total": len(decs), "n_live_decisions": len(live),
                   "kl": kl_sum / n_all, "entropy": ent_sum / n_all, "legal_mass": sum(masses) / max(1, len(masses)),
                   "grad_norm": gnorm}
            for k in TELEMETRY_KEYS:
                row[k] = sum(float(o[k]) for o in outs) / len(outs)
            tf.write(json.dumps(row) + "\n"); tf.flush()
            bar.update(1); bar.set_postfix(r=f"{row['reward_mean']:.2f}", win=f"{row['win']:.2f}", ko=f"{row['ko']:.2f}", kl=f"{row['kl']:.3f}")
    model.eval()
    return {"steps_done": done, "stopped": stopped, "minutes": (time.time() - t_start) / SECONDS_PER_MINUTE}
