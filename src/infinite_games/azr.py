"""Absolute Zero for games: one model proposes games as code and plays them, trained on both roles at once.

One training step:
    1. PROPOSE   ``proposals`` games, each prompted with the interface and ``n_refs`` games sampled from the buffer
                 (seeded with the hand-written training seeds), asked for a new, somewhat more complex game.
    2. VALIDATE  in sandboxed worker processes (azr_sandbox.validate): no code -> -1, fails a check -> -0.5.
    3. PLAY      in one lockstep batch through the GameServer, with the LLM choosing moves by letter:
                 learnability instances  every valid proposal x ``learn_instances``, seats [LLM, LLM, novice]
                                         (novice = uniform random move, seat rotated): success = the LLM seats'
                                         mean take-out exceeds the novice's;
                 economy instances       ``economy_games`` games (valid proposals first, then buffer games) x
                                         ``group`` replays, all seats LLM, entered from a persistent world's
                                         wallets; the world advances with replay 0 and restarts on ruin / stomp.
    4. REWARD    proposer: learnability(success) x agency x temptation (azr_economy.proposer_reward);
                 solver: azr_economy.rewards(arm) per seat.
    5. UPDATE    one AdamW step on proposer loss (REINFORCE, batch-mean baseline, token-mean log-prob, k3 KL to the
                 base) + solver loss (letter policy, per-seat group baseline, exact KL); the control run drops the
                 solver loss and plays with the adapter disabled (solver frozen at base).
Valid proposals join the buffer. Everything is logged per step (JSONL) and every proposal is kept (proposals.jsonl).
"""
from __future__ import annotations

import contextlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from . import azr_economy as EC
from . import azr_sandbox as SB
from . import engine as E
from . import policy as PO
from .azr_text import (CODE_RE, LLM, NOVICE, PROPOSER_SYSTEM, extract_code, novice_seats,  # noqa: F401
                       parse_reflection, proposer_prompt, reflect_prompt, rules_of, solver_prompt, RESTRAINED,
                       acts_on_thought, restrained_moves)

SECONDS_PER_MINUTE = 60.0
TRACE_DESCRIBE_CHARS = 400
SAY_FALLBACK_TOKENS = 40


# ---------------------------------------------------------------------------------------------------- proposing
def chat_ids(tok, user: str, system: Optional[str] = None) -> List[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string=PO.TEMPLATE_DATE)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generate(model, tok, prompt_ids: Sequence[List[int]], max_new: int, temperature: float, seed: int, batch: int) -> List[List[int]]:
    """Sampled completions (token ids, without prompt or padding) for each prompt."""
    torch.manual_seed(seed); outs: List[List[int]] = []; pad = tok.pad_token_id; eos = tok.eos_token_id
    model.eval()
    for s in range(0, len(prompt_ids), batch):
        chunk = prompt_ids[s:s + batch]; T = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long); att = torch.zeros_like(ids)
        for r, x in enumerate(chunk):
            ids[r, T - len(x):] = torch.tensor(x); att[r, T - len(x):] = 1
        dev = next(model.parameters()).device
        gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, do_sample=True,
                             temperature=temperature, top_p=1.0, pad_token_id=pad, use_cache=True)
        for row in gen[:, T:].tolist():
            cut = next((i for i, t in enumerate(row) if t == eos or t == pad), len(row)); outs.append(row[:cut + 1] if cut < len(row) else row)
    return outs


def completion_logprob(model, prompt: List[int], comp: List[int], ref: bool = False) -> torch.Tensor:
    """Per-token log-probs of ``comp`` after ``prompt`` (grad unless ref)."""
    dev = next(model.parameters()).device; ids = torch.tensor([prompt + comp], device=dev); keep = len(comp) + 1
    if ref:
        with torch.no_grad(), model.disable_adapter():
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits
    else:
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits
    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)          # the positions that predict each completion token
    return lp.gather(1, torch.tensor(comp, device=dev)[:, None]).squeeze(1)


def batch_completion_logprob(model, prompts: Sequence[List[int]], comps: Sequence[List[int]], pad: int, ref: bool = False) -> List[torch.Tensor]:
    """completion_logprob for a batch: left-padded so every completion ends at the last position, logits kept only for
    the longest completion (+1); returns per-row token log-probs. Rotary positions make the left offset harmless and
    the attention mask hides the padding (the same convention as policy.LetterPolicy)."""
    dev = next(model.parameters()).device; seqs = [list(p) + list(c) for p, c in zip(prompts, comps)]
    T = max(len(x) for x in seqs); keep = max(len(c) for c in comps) + 1
    ids = torch.full((len(seqs), T), pad, dtype=torch.long); att = torch.zeros_like(ids)
    for r, x in enumerate(seqs):
        ids[r, T - len(x):] = torch.tensor(x); att[r, T - len(x):] = 1
    ids, att = ids.to(dev), att.to(dev)
    if ref:
        with torch.no_grad(), model.disable_adapter():
            logits = model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=keep).logits
    else:
        logits = model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=keep).logits
    out = []
    for r, c in enumerate(comps):
        lp = torch.log_softmax(logits[r, keep - len(c) - 1:-1].float(), dim=-1)
        out.append(lp.gather(1, torch.tensor(c, device=dev)[:, None]).squeeze(1))
    return out


# ------------------------------------------------------------------------------------------------------- playing
def play_batch(server: SB.GameServer, pol: PO.LetterPolicy, instances: List[Dict], rng: random.Random,
               gen: torch.Generator, ref: bool, desc: Optional[str] = None, think: Optional[Dict] = None) -> Tuple:
    """Play every instance to the end in lockstep. instance: {code, rules, endowments, bystanders, seed, seats, world}.

    With ``think`` ({every, max_new, batch, temperature}) an LLM player THINKS at its first turn and every ``every``-th
    own turn after: it writes private reasoning and one public message (azr_text.reflect_prompt); the message joins the
    instance's public message list, and every later move prompt shows the world, the messages and the player's own
    latest thinking. Returns (results, decisions, masses, reflections, traces):
        decisions    (instance, seat, prompt, n options, choice) of every LLM move
        reflections  {inst, seat, turn, prompt_ids, comp_ids, think, say} of every thinking step (trained like moves)
        traces       per instance, every move in order: {turn, player, seat, move, n_moves, think, say, describe}"""
    res: List[Dict] = [{} for _ in instances]; decisions: List[Tuple] = []; masses: List[float] = []
    reflections: List[Dict] = []; traces: List[List[Dict]] = [[] for _ in instances]
    msgs: List[List[str]] = [[] for _ in instances]; thought: Dict[Tuple[int, int], str] = {}; own = Counter()
    last_ref: Dict[Tuple[int, int], int] = {}
    obs = server.step({i: {"op": "new", "code": x["code"], "n": SB.N_PLAYERS, "endowments": x["endowments"],
                           "bystanders": x["bystanders"], "seed": x["seed"]} for i, x in enumerate(instances)})
    live = {}
    for i, ob in obs.items():
        if "error" in ob:
            res[i] = {"error": ob["error"], "detail": ob.get("detail", "")}
        elif ob["over"]:
            res[i] = {"out": ob, "turns": 0}
        else:
            live[i] = ob
    turns = Counter(); bar = tqdm(desc=desc, unit="turn", leave=False, disable=desc is None)
    world = lambda i: instances[i].get("world", "")
    while live:
        ops, llm = {}, []
        for i, ob in live.items():
            seat = instances[i]["seats"][ob["player"]]
            if turns[i] >= SB.MAX_TURNS:
                res[i] = {"error": "too_long_game", "detail": "live"}; continue
            if seat in (NOVICE, RESTRAINED):
                mv = rng.choice(ob["moves"] if seat == NOVICE else restrained_moves(ob["moves"]))
                ops[i] = {"op": "play", "player": ob["player"], "move": mv}
                traces[i].append({"turn": turns[i], "player": ob["player"], "seat": seat, "move": mv, "n_moves": len(ob["moves"])})
            else:
                llm.append(i)
        if llm and think:
            need = [i for i in llm if own[(i, live[i]["player"])] % int(think["every"]) == 0]
            if need:
                texts = [reflect_prompt(instances[i]["rules"], live[i]["describe"], live[i]["moves"], live[i]["player"], world(i), msgs[i]) for i in need]
                ids = [chat_ids(pol.tok, t) for t in texts]
                with (pol.model.disable_adapter() if ref and hasattr(pol.model, "disable_adapter") else contextlib.nullcontext()):
                    comps = generate(pol.model, pol.tok, ids, int(think["max_new"]), float(think.get("temperature", 1.0)),
                                     rng.randrange(2 ** 31), int(think["batch"]))
                miss = [n for n, c in enumerate(comps) if "SAY:" not in pol.tok.decode(c, skip_special_tokens=True)]
                if miss:                                           # the message never came: ask for it (short, one wave)
                    say_ids = pol.tok("\nSAY:", add_special_tokens=False)["input_ids"]
                    cut = [[t for t in comps[n] if t not in (pol.tok.eos_token_id, pol.tok.pad_token_id)] + say_ids for n in miss]
                    with (pol.model.disable_adapter() if ref and hasattr(pol.model, "disable_adapter") else contextlib.nullcontext()):
                        more = generate(pol.model, pol.tok, [ids[n] + c for n, c in zip(miss, cut)], SAY_FALLBACK_TOKENS,
                                        float(think.get("temperature", 1.0)), rng.randrange(2 ** 31), int(think["batch"]))
                    for n, c, m in zip(miss, cut, more):
                        comps[n] = c + m
                for i, pid, cid in zip(need, ids, comps):
                    p = live[i]["player"]; r = parse_reflection(pol.tok.decode(cid, skip_special_tokens=True))
                    thought[(i, p)] = r["think"] + (f"\nI told the others: {r['say']}" if r["say"] else "")
                    if r["say"]:
                        msgs[i].append(f"P{p} (turn {turns[i]}): {r['say']}")
                    reflections.append({"inst": i, "seat": p, "turn": turns[i], "prompt_ids": pid, "comp_ids": cid, "n_moves": 0, "n_agree": 0, **r})
                    last_ref[(i, p)] = len(reflections) - 1
        if llm:
            prompts = [solver_prompt(instances[i]["rules"], live[i]["describe"], live[i]["moves"], live[i]["player"], world(i), msgs[i],
                                     thought.get((i, live[i]["player"]), "")) for i in llm]
            probs = pol.probs(prompts, [len(live[i]["moves"]) for i in llm], ref=ref); masses += pol.last_legal_mass
            for i, pr, pt in zip(llm, probs, prompts):
                p = live[i]["player"]; k = PO.sample(pr, gen); decisions.append((i, p, pt, len(pr), k)); own[(i, p)] += 1
                mv = live[i]["moves"][k]; ops[i] = {"op": "play", "player": p, "move": mv}
                ri = last_ref.get((i, p))
                if ri is not None:                                 # does the move follow the thought it came after?
                    reflections[ri]["n_moves"] += 1; reflections[ri]["n_agree"] += int(acts_on_thought(mv, reflections[ri]["think"]))
                last = next((r for r in reversed(reflections) if r["inst"] == i and r["seat"] == p and r["turn"] == turns[i]), None) if think else None
                traces[i].append({"turn": turns[i], "player": p, "seat": LLM, "move": mv, "n_moves": len(pr), "p_move": pr[k],
                                  "think": last["think"] if last else None, "say": last["say"] if last else None,
                                  "describe": live[i]["describe"][:TRACE_DESCRIBE_CHARS]})
        live = {i: ob for i, ob in live.items() if i in ops}
        if not ops:
            break
        for i, ob in server.step(ops).items():
            turns[i] += 1
            if "error" in ob:
                res[i] = {"error": ob["error"], "detail": ob.get("detail", "")}; live.pop(i, None)
            elif ob["over"]:
                res[i] = {"out": ob, "turns": turns[i]}; live.pop(i, None)
            else:
                live[i] = ob
        bar.update(len(ops))
    bar.close()
    server.step({i: {"op": "drop"} for i in range(len(instances))})
    return res, decisions, masses, reflections, traces


# ---------------------------------------------------------------------------------------------------------- warm start
def sft_pairs(tok, seeds: Sequence[Dict], n: int, n_refs: int, rng: random.Random, max_ref_chars: int) -> List[Tuple[List[int], List[int]]]:
    """Supervised examples for the warm start: the proposer prompt with ``n_refs`` seed games as examples, and as the target
    a DIFFERENT seed game in a code block, so the model learns the interface and the form, not a memorised copy."""
    short = [s for s in seeds if len(s["code"]) <= max_ref_chars] or list(seeds); out = []
    for _ in range(n):
        target = seeds[rng.randrange(len(seeds))]; pool = [s for s in short if s["code"] != target["code"]] or short
        refs = [pool[rng.randrange(len(pool))]["code"] for _ in range(n_refs)]
        prompt = chat_ids(tok, proposer_prompt(refs), PROPOSER_SYSTEM)
        comp = tok("```python\n" + target["code"].strip() + "\n```", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        out.append((prompt, comp))
    return out


def sft_warmup(model, tok, pairs: Sequence[Tuple[List[int], List[int]]], cfg: Dict, log_path: Path) -> Dict:
    """Cross-entropy on the target tokens (LoRA only), micro-batched; logged per optimizer step."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0.0); bs = int(cfg["batch"]); losses = []
    model.train(); log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        for ep in range(int(cfg["epochs"])):
            for s0 in tqdm(range(0, len(pairs), bs), desc=f"warm start epoch {ep + 1}", unit="batch"):
                mb = pairs[s0:s0 + bs]; opt.zero_grad(set_to_none=True)
                lps = batch_completion_logprob(model, [p for p, _ in mb], [c for _, c in mb], tok.pad_token_id)
                loss = -sum(lp.mean() for lp in lps) / len(lps); loss.backward()
                g = float(torch.nn.utils.clip_grad_norm_(params, float(cfg.get("grad_clip", 1.0)))); opt.step()
                losses.append(float(loss)); f.write(json.dumps({"epoch": ep, "batch": s0 // bs, "nll": float(loss), "grad_norm": g}) + "\n"); f.flush()
    model.eval()
    k = max(1, len(losses) // 10)
    return {"batches": len(losses), "nll_first": sum(losses[:k]) / k, "nll_last": sum(losses[-k:]) / k}


# ---------------------------------------------------------------------------------------------------------- run
class Buffer:
    def __init__(self, seeds: Sequence[Dict], max_ref_chars: Optional[int] = None):
        self.items: List[Dict] = [dict(s, source="seed") for s in seeds]; self.max_ref_chars = max_ref_chars

    def sample(self, k: int, rng: random.Random, refs: bool = False) -> List[Dict]:
        """``refs``: prefer games no longer than max_ref_chars as proposer examples (short examples, short proposals)."""
        pool = self.items
        if refs and self.max_ref_chars:
            pool = [it for it in self.items if len(it["code"]) <= self.max_ref_chars] or self.items
        return [pool[rng.randrange(len(pool))] for _ in range(k)]

    def add(self, item: Dict) -> None:
        self.items.append(item)


def train_run(model, tok, pol: PO.LetterPolicy, seeds: Sequence[Dict], cfg: Dict, arm: str, joint: bool, seed: int,
              out_dir: Path, n_steps: int, max_minutes: Optional[float], pool: SB.Pool, server: SB.GameServer,
              conditions: Optional[Dict[str, bool]] = None, inject: Optional[str] = None) -> Dict:
    """One AZR run. ``joint`` = solver trains too (the finite / infinite runs); False = proposer only, solver = base.
    ``conditions`` switches the terms of azr_economy.CONDITIONS (missing = on): the four solver penalties (infinite
    arm), the three proposer factors, and resource carryover between games."""
    cond = {c: True for c in EC.CONDITIONS}; cond.update(conditions or {})
    bad = set(cond) - set(EC.CONDITIONS)
    if bad:
        raise ValueError(f"unknown conditions {sorted(bad)}")
    weights = {k: (EC.DEFAULT_WEIGHTS[k] if cond[EC.WEIGHT_CONDITION[k]] else 0.0) for k in EC.DEFAULT_WEIGHTS}
    restore_prob, bitter_games = float(cfg["stomp_restore_prob"]), int(cfg["bitter_games"])
    out_dir.mkdir(parents=True, exist_ok=True); rng = random.Random(seed); buf = Buffer(seeds, cfg.get("max_ref_chars"))
    worlds = [EC.World(seed=seed * 1000 + k) for k in range(int(cfg["worlds"]))]
    if len(worlds) < int(cfg["economy_games"]):
        raise ValueError("worlds must be >= economy_games (each game in a step needs its own world)")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0.0)
    P, G = int(cfg["proposals"]), int(cfg["group"]); t0 = time.time(); done = 0; stopped = "steps"
    tel = open(out_dir / "train_telemetry.jsonl", "a"); props = open(out_dir / "proposals.jsonl", "a")
    bar = tqdm(total=n_steps, desc=f"azr {arm}{'' if joint else '-control'} s{seed}", unit="step")
    for step in range(n_steps):
        if max_minutes is not None and (time.time() - t0) / SECONDS_PER_MINUTE > float(max_minutes):
            stopped = "time"; break
        ts = time.time(); gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
        # 1-2: propose and validate
        refs = [[r["code"] for r in buf.sample(int(cfg["n_refs"]), rng, refs=True)] for _ in range(P)]
        prompts = [chat_ids(tok, proposer_prompt(rf), PROPOSER_SYSTEM) for rf in refs]
        comps = generate(model, tok, prompts, int(cfg["max_new_tokens"]), float(cfg["temperature"]), seed * 7919 + step, int(cfg["gen_batch"]))
        texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
        codes = [extract_code(t) for t in texts]; verdicts: List[Optional[Dict]] = [None] * P
        if inject and step == 0:
            codes[0] = inject                                       # smoke: one known-valid game through the valid path
        known = {_norm(it["code"]) for it in buf.items}
        for i, c in enumerate(codes):
            if c is not None and (not inject or step or i) and _norm(c) in known:
                verdicts[i] = {"ok": False, "reason": "copy", "detail": "a game already in the buffer, verbatim"}
        todo = [i for i, c in enumerate(codes) if c is not None and verdicts[i] is None]
        for i, v in zip(todo, pool.validate_many([codes[i] for i in todo], seed=step)):
            verdicts[i] = v
        for i in range(P):
            if verdicts[i] is None:
                verdicts[i] = {"ok": False, "reason": "no_code", "detail": ""}
        valid = [i for i in range(P) if verdicts[i]["ok"]]
        # 3: play (learnability + economy) in one lockstep batch
        inst: List[Dict] = []; meta: List[Tuple] = []
        for i in valid:
            for k in range(int(cfg["learn_instances"])):
                inst.append({"code": codes[i], "rules": verdicts[i]["stats"]["rules"], "endowments": [SB.DEFAULT_ENDOWMENT] * SB.N_PLAYERS,
                             "bystanders": [SB.DEFAULT_BYSTANDER] * SB.N_BYSTANDERS, "seed": rng.randrange(2 ** 31), "seats": novice_seats(k),
                             "world": EC.world_text(None)})
                meta.append(("learn", i, k))
        games = [(codes[i], verdicts[i]["stats"]["rules"], "proposal", i) for i in valid][:int(cfg["economy_games"])]
        for g in buf.sample(max(0, int(cfg["economy_games"]) - len(games)), rng):
            games.append((g["code"], g["rules"], g.get("source", "buffer"), None))
        before = []
        for j, (code, rules, src, pi) in enumerate(games):
            w = worlds[(step * len(games) + j) % len(worlds)]; endow = w.endowments(); before.append((w, endow, EC.World(**vars(w))))
            gs = rng.randrange(2 ** 31); wt = EC.world_text(w, cond["carryover"])
            for r in range(G):
                inst.append({"code": code, "rules": rules, "endowments": list(endow), "bystanders": list(w.bystanders),
                             "seed": gs + r, "seats": [LLM] * SB.N_PLAYERS, "world": wt})
                meta.append(("econ", j, r))
        think = cfg.get("think") if cfg.get("think", {}).get("enabled", True) else None
        results, decs, masses, refl, traces = play_batch(server, pol, inst, rng, gen, ref=not joint, think=think) if inst else ([], [], [], [], [])
        # 4: rewards
        succ: Dict[int, List[float]] = {i: [] for i in valid}; broken = set()
        for (kind, a, b), x, res in zip(meta, inst, results):
            if kind == "learn":
                if "error" in res:
                    broken.add(a); continue
                r = res["out"]["resources"]; nov = x["seats"].index(NOVICE)
                llm_mean = sum(v for s, v in enumerate(r) if s != nov) / (SB.N_PLAYERS - 1)
                succ[a].append(1.0 if llm_mean > r[nov] else (0.5 if llm_mean == r[nov] else 0.0))
        p_rew = []
        for i in range(P):
            if i in broken:
                verdicts[i] = {"ok": False, "reason": "runtime_in_play", "detail": "", "stats": verdicts[i].get("stats")}
            sr = sum(succ.get(i) or [0.0]) / max(1, len(succ.get(i) or []))
            p_rew.append(EC.proposer_reward(verdicts[i], sr, cond["learnability"], cond["agency"], cond["temptation"]))
            verdicts[i]["success_rate"] = sr if verdicts[i]["ok"] else None
        s_rew: Dict[int, List[float]] = {}; econ = []
        for j in range(len(games)):
            w, endow, snap = before[j]; outs = {b: res for (kind, a, b), res in zip(meta, results) if kind == "econ" and a == j}
            if any("error" in o for o in outs.values()) or len(outs) != G:
                continue
            for r in range(G):
                ev = EC.game_outcome(snap, outs[r]["out"], bitter_games); s_rew[(j, r)] = EC.rewards(arm, endow, ev, weights)
            ev0 = EC.settle(w, endow, outs[0]["out"], carryover=cond["carryover"], restore_prob=restore_prob, bitter_games=bitter_games)
            econ.append(dict(ev0, source=games[j][2]))
        # 5: update
        model.train(); opt.zero_grad(set_to_none=True); base_p = sum(p_rew) / P; kl_p = kl_s = 0.0; n_p = 0
        for i in range(P):
            adv = p_rew[i] - base_p
            if abs(adv) < 1e-9 or not comps[i]:
                continue
            lp = completion_logprob(model, prompts[i], comps[i]); lr_ = completion_logprob(model, prompts[i], comps[i], ref=True)
            d = lr_ - lp; kl = (torch.exp(d) - d - 1).mean()
            ((-adv * lp.mean() + float(cfg["beta_proposer"]) * kl) / P).backward(); kl_p += float(kl); n_p += 1
        grp = {}
        idx = {m: n for n, m in enumerate(meta)}
        for j in range(len(games)):
            if (j, 0) not in s_rew:
                continue
            for seat in range(SB.N_PLAYERS):
                vals = [s_rew[(j, r)][seat] for r in range(G)]; mu = sum(vals) / G
                for r in range(G):
                    grp[(idx[("econ", j, r)], seat)] = vals[r] - mu
        sd = [d for d in decs if (d[0], d[1]) in grp] if joint else []
        n_all = max(1, len(sd))
        for s in range(0, len(sd), pol.micro_batch):
            mb = sd[s:s + pol.micro_batch]; cur = pol.forward([d[2] for d in mb], [d[3] for d in mb])
            with torch.no_grad():
                refp = pol.forward([d[2] for d in mb], [d[3] for d in mb], ref=True)
            ch = torch.tensor([d[4] for d in mb], device=cur["logp"].device)
            a = torch.tensor([grp[(d[0], d[1])] for d in mb], device=cur["logp"].device, dtype=cur["logp"].dtype)
            ke = PO.kl_and_entropy(cur["logp"], refp["logp"])
            ((-(a * cur["logp"].gather(1, ch[:, None]).squeeze(1)) + float(cfg["beta_solver"]) * ke["kl"]).sum() / n_all).backward()
            kl_s += float(ke["kl"].sum())
        tr = [r for r in refl if (r["inst"], r["seat"]) in grp and r["comp_ids"]] if joint else []
        kl_t = 0.0; tb = int(cfg["think"]["train_batch"]) if tr else 1
        cons = [r["n_agree"] / r["n_moves"] if r["n_moves"] else 0.0 for r in tr]; mc = sum(cons) / len(cons) if cons else 0.0
        tw, cw = float(cfg["think"].get("loss_weight", 1.0)), float(cfg["think"].get("consistency_weight", 0.0))
        for r, c in zip(tr, cons):                                 # seat advantage + a bonus for moves that follow the thought
            r["adv"] = grp[(r["inst"], r["seat"])] + cw * (c - mc)
        for s0 in range(0, len(tr), tb):                           # thinking + message tokens get the seat's advantage too
            mb = tr[s0:s0 + tb]; P_, C_ = [r["prompt_ids"] for r in mb], [r["comp_ids"] for r in mb]
            lps = batch_completion_logprob(model, P_, C_, tok.pad_token_id); lrs = batch_completion_logprob(model, P_, C_, tok.pad_token_id, ref=True)
            loss = 0.0
            for r, lp, lr_ in zip(mb, lps, lrs):
                d = lr_ - lp; kl = (torch.exp(d) - d - 1).mean(); kl_t += float(kl)
                loss = loss + tw * (-r["adv"] * lp.mean() + float(cfg["beta_solver"]) * kl) / len(tr)
            loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))); opt.step(); model.eval(); done = step + 1
        with open(out_dir / "traces.jsonl", "a") as tf:             # replay 0 of every economy game: the one the world keeps
            for j in range(len(games)):
                n = idx.get(("econ", j, 0))
                if n is None or "error" in results[n]:
                    continue
                o = results[n]["out"]; w, endow, snap = before[j]
                tf.write(json.dumps({"step": step, "game": j, "source": games[j][2], "arm": arm, "seed": seed,
                                     "world_before": {"wallets": snap.wallets, "bystanders": snap.bystanders, "epoch_games": snap.epoch_games,
                                                      "bitter_left": snap.bitter_left},
                                     "rules": games[j][1][:1500], "moves": traces[n], "resources": o["resources"], "bystanders_after": o["bystanders"],
                                     "rewards": s_rew.get((j, 0))}) + "\n")
        # buffer, logs
        for i in range(P):
            rec = {"step": step, "i": i, "reason": None if verdicts[i]["ok"] else verdicts[i]["reason"], "detail": verdicts[i].get("detail", "")[:300],
                   "reward": p_rew[i], "success_rate": verdicts[i].get("success_rate"), "code": codes[i],
                   "rules": (verdicts[i].get("stats") or {}).get("rules") or rules_of(codes[i]),
                   "text": texts[i] if codes[i] is None else None,
                   "stats": {k: v for k, v in (verdicts[i].get("stats") or {}).items() if k != "rules"}}
            props.write(json.dumps(rec) + "\n")
            if verdicts[i]["ok"]:
                buf.add({"code": codes[i], "rules": verdicts[i]["stats"]["rules"], "source": f"s{seed}_step{step}", "stats": rec["stats"]})
        props.flush()
        vst = [verdicts[i]["stats"] for i in valid]
        row = {"step": step, "arm": arm, "joint": joint, "seed": seed, "seconds": round(time.time() - ts, 1),
               "valid_share": len(valid) / P, "reasons": dict(Counter(v["reason"] for v in verdicts if not v["ok"])),
               "proposer_reward": base_p, "success_rate": _mean([verdicts[i].get("success_rate") for i in valid]),
               "agency": _mean([s["agency"] for s in vst]), "bystander_temptation": _mean([float(s["bystander_temptation"]) for s in vst]),
               "ruin_temptation": _mean([float(s["ruin_temptation"]) for s in vst]), "code_lines": _mean([s["code_lines"] for s in vst]),
               "distinct_moves": _mean([s["distinct_moves"] for s in vst]), "legal_per_turn": _mean([s["legal_per_turn"] for s in vst]),
               "turns": _mean([s["turns"] for s in vst]), "buffer": len(buf.items), "economy_games": len(econ),
               "restarts": sum(e["restarted"] for e in econ), "ruins": sum(e["ruin"] for e in econ), "stomps": sum(e["stomp"] for e in econ),
               "restores": sum(e["restored"] for e in econ), "bitter_world_games": sum(e["stomped_before"] for e in econ),
               "bystander_loss": _mean([e["bystander_loss"] for e in econ]), "top_share": _mean([e["top_share"] for e in econ]),
               "solver_reward": _mean([v for vs in s_rew.values() for v in vs]), "solver_decisions": len(sd), "llm_decisions": len(decs),
               "kl_proposer": kl_p / max(1, n_p), "kl_solver": kl_s / n_all, "kl_think": kl_t / max(1, len(tr)), "grad_norm": gnorm,
               "reflections": len(refl), "reflections_trained": len(tr), "messages_sent": sum(bool(r["say"]) for r in refl),
               "think_tokens": _mean([len(r["comp_ids"]) for r in refl]), "think_consistency": _mean([r["n_agree"] / r["n_moves"] for r in refl if r["n_moves"]]),
               "legal_mass": _mean(masses), "gen_tokens": _mean([len(c) for c in comps]),
               "world_gini": _mean([EC.gini(w.wallets) for w in worlds]), "world_restarts_total": sum(w.restarts for w in worlds)}
        tel.write(json.dumps(row) + "\n"); tel.flush()
        bar.update(1); bar.set_postfix(valid=f"{row['valid_share']:.2f}", pr=f"{base_p:.2f}", rst=row["restarts"], sr=row["success_rate"])
    bar.close(); tel.close(); props.close()
    with open(out_dir / "buffer.jsonl", "w") as f:
        for it in buf.items:
            f.write(json.dumps({k: it[k] for k in ("code", "rules", "source") if k in it}) + "\n")
    with open(out_dir / "worlds.json", "w") as f:
        json.dump([vars(w) for w in worlds], f)
    return {"steps_done": done, "stopped": stopped, "minutes": (time.time() - t0) / SECONDS_PER_MINUTE, "buffer": len(buf.items)}


def _norm(code: str) -> str:
    return "".join(code.split())


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None
