"""Absolute Zero for games, night 3: ONE model holds three roles behind latches -- the proposer, player type A and player
type B -- and trains all three at once (Thomas 2026-09-29).

    proposer   system "You are the proposer." Writes games as sandboxed Python; reward = STAGE_REWARD if invalid, else
               VALID_BONUS + gate x QUALITY_MAX x mean(skill, agency, temptation, intervention) (azr_economy.proposer_reward3);
               the gate opens with the recent valid rate, so validity comes first and better games after. Skill = the better
               of two contests: base vs a random player, and the CURRENT model vs a PAST snapshot of itself.
    type A     system "You are player type A." Rewarded by the rule the run gives A (finite in every run tonight).
    type B     system "You are player type B." Rewarded by the rule the run gives B (infinite; infinite_own in the
               own-harm control, which charges B only for its own share of the harm).

Economy games come in four compositions (AAA, BBB, ABB, AAB; equal shares): single games, and joint games where each seat
is rewarded by its own rule, so an infinite B pays for what a finite A does to the bystanders. Worlds are persistent per
composition (seats keep their types for a world's life). Each game draws whether the players' types are public.
Every move's effect is read from the game (final_resources / final_bystanders after each move) and attributed to the
player who moved: that feeds the own-harm rule and the traces.

Training is LoRA or full fine-tuning (``train.mode``); ``Nets`` switches between the current weights, the base (the KL
reference and the base contest) and the past snapshot (the other contest) without a second copy of the base in LoRA mode.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from . import azr as AZ
from . import azr_economy as EC
from . import azr_sandbox as SB
from . import azr_sanction as SN
from . import policy as PO
from .azr_text import (LLM, NOVICE, PROPOSER_SYSTEM3, RAIDER, RESTRAINED, acts_on_thought, latch, move_verb, novice_seats,
                       parse_reflection, proposer_prompt3, raider_moves, reflect_prompt, restrained_moves, roles_text, rules_of,
                       solver_prompt)

SECONDS_PER_MINUTE = 60.0
TRACE_DESCRIBE_CHARS = 400
SAY_FALLBACK_TOKENS = 40
WHO = ("cur", "base", "past")
MODES = ("none", "lora", "full")
COMPOSITIONS = ("AAA", "BBB", "ABB", "AAB")
PAST_ADAPTER = "past"
DEFAULT_ADAPTER = "default"


# ------------------------------------------------------------------------------------------------------ networks
ROLE_ADAPTERS = {"A": "role_A", "B": "role_B", "P": "role_P", "W": "role_W", "R": "role_R", "Q": "role_Q"}   # night 4: one LoRA per role ("A"/"B" would collide with lora_A/lora_B);
# W (azr12, 2026-10-14): pure winners in the voting game, created only when train.extra_roles lists it (base roles A, B, P always)
BASE_ROLES = ("A", "B", "P")
# R (azr14): a frozen copy of the warm-started proposer, the proposer's KL reference; never trained (extra_roles)
SNAP_ROLE = "A"                                                       # the past-self opponent is type A's own past (contests play as A)


def split_who(who: str) -> Tuple[str, Optional[str]]:
    """'cur:B' -> ('cur', 'B'); 'base' -> ('base', None)."""
    kind, _, role = who.partition(":")
    if kind not in WHO or (role and role not in ROLE_ADAPTERS):
        raise ValueError(f"who {who!r}")
    return kind, role or None


class Nets:
    """Which weights answer: ``cur`` (being trained), ``base`` (the untrained model: KL reference, base contest) or ``past``
    (the oldest snapshot in the ring; the base until the first snapshot). ``cur:A`` / ``cur:B`` / ``cur:P`` name a role.

    mode none  evaluation of the untrained model: every role is the base
    mode lora  one PEFT model: base = adapters disabled, past = an extra adapter loaded from disk (no second base copy).
               ``roles`` (night 4) = {role: adapter name}: each role has ITS OWN adapter on the shared frozen base, so the
               finite and infinite rewards no longer act on the same weights; None = one shared adapter (night 3)
    mode full  ``model`` is a float32 copy trained with bf16 autocast; ``base_model`` the frozen original; the past
               snapshot is a third (bf16) copy loaded from disk (one set of weights for every role)"""

    def __init__(self, pol: PO.LetterPolicy, mode: str, model, base_model=None, roles: Optional[Dict[str, str]] = None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} (valid: {MODES})")
        if roles and mode != "lora":
            raise ValueError("separate role adapters need mode lora")
        self.pol, self.mode, self.model, self.base_model, self.roles = pol, mode, model, base_model, roles
        self.past_model = None; self.past_ready = False; self.past_step: Optional[int] = None; self.ring: List[Tuple[int, Path]] = []

    def adapter_of(self, role: Optional[str]) -> str:
        return self.roles[role or SNAP_ROLE] if self.roles else DEFAULT_ADAPTER

    @contextlib.contextmanager
    def use(self, who: str):
        kind, role = split_who(who)
        prev = self.pol.model
        try:
            if self.mode == "none":
                self.pol.model = self.model; yield self.model
            elif self.mode == "lora":
                if kind == "cur":
                    if self.roles:
                        self.model.set_adapter(self.adapter_of(role))
                    self.pol.model = self.model; yield self.model
                elif kind == "past" and self.past_ready:
                    back = self.model.active_adapter
                    self.model.set_adapter(PAST_ADAPTER)
                    try:
                        self.pol.model = self.model; yield self.model
                    finally:
                        self.model.set_adapter(back if isinstance(back, str) else back[0])
                else:
                    with self.model.disable_adapter():
                        self.pol.model = self.model; yield self.model
            else:
                if kind == "cur":
                    dev = next(self.model.parameters()).device
                    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        self.pol.model = self.model; yield self.model
                else:
                    m = self.past_model if (kind == "past" and self.past_ready) else self.base_model
                    self.pol.model = m; yield m
        finally:
            self.pol.model = prev

    def trainable(self) -> List[torch.nn.Parameter]:
        if self.roles:                       # set_adapter freezes the inactive adapters, so collect every role's LoRA by name
            return [p for n, p in self.model.named_parameters() if "lora_" in n and any(f".{a}." in n for a in self.roles.values())]
        return [p for p in self.model.parameters() if p.requires_grad]

    def role_params(self) -> Dict[str, List[torch.nn.Parameter]]:
        """{role: its adapter's parameters} (one group 'all' for a shared adapter)."""
        if not self.roles:
            return {"all": self.trainable()}
        return {r: [p for n, p in self.model.named_parameters() if "lora_" in n and f".{a}." in n] for r, a in self.roles.items()}

    def snapshot(self, step: int, root: Path, ring: int) -> None:
        """Save the current weights (type A's adapter when roles are separate), keep the newest ``ring`` snapshots, and load
        the OLDEST as the past opponent."""
        if self.mode == "none":
            return
        d = root / f"step_{step:04d}"; d.mkdir(parents=True, exist_ok=True)
        if self.mode == "lora":
            name = self.adapter_of(SNAP_ROLE); self.model.save_pretrained(str(d), selected_adapters=[name])
        else:
            torch.save({k: v.detach().to(torch.bfloat16).cpu() for k, v in self.model.state_dict().items()}, d / "state.pt")
        self.ring.append((step, d))
        while len(self.ring) > ring:
            _, old = self.ring.pop(0); shutil.rmtree(old, ignore_errors=True)
        step0, d0 = self.ring[0]
        if self.mode == "lora":
            name = self.adapter_of(SNAP_ROLE); src = d0 / name if (d0 / name / "adapter_config.json").exists() else d0
            back = self.model.active_adapter
            if PAST_ADAPTER in self.model.peft_config:
                self.model.delete_adapter(PAST_ADAPTER)
            self.model.load_adapter(str(src), adapter_name=PAST_ADAPTER, is_trainable=False)
            self.model.set_adapter(back if isinstance(back, str) else back[0])
        else:
            if self.past_model is None:
                self.past_model = copy.deepcopy(self.base_model)
            sd = torch.load(d0 / "state.pt", map_location="cpu")
            self.past_model.load_state_dict({k: v.to(next(self.past_model.parameters()).dtype) for k, v in sd.items()})
        self.past_ready, self.past_step = True, step0


# -------------------------------------------------------------------------------------------------------- playing
def _attrib0(n: int) -> Dict:
    return {"bys_by": [0] * n, "loss_by": [[0] * n for _ in range(n)], "moves_by": [0] * n}


def play_batch3(server: SB.GameServer, pol: PO.LetterPolicy, nets: Nets, instances: List[Dict], rng: random.Random,
                gen: torch.Generator, desc: Optional[str] = None, think: Optional[Dict] = None) -> Tuple:
    """Play every instance to the end in lockstep (the night-3 counterpart of azr.play_batch).

    instance: {code, rules, endowments, bystanders, seed, seats, types, who, public, world}
        seats  per seat: LLM, NOVICE (uniform random), RESTRAINED (random, never names a party), RAIDER (takes from a
               bystander whenever it can)
        types  per seat 'A' / 'B' / None: an LLM seat's latch, and what a public game shows about every seat
        who    per seat 'cur' / 'base' / 'past' (LLM seats): which weights play it
    Returns (results, decisions, masses, reflections, traces); every result carries ``attrib`` (the per-move readouts:
    bystander holdings lost on each seat's moves, each player's holdings lost on each seat's moves); decisions are
    (inst, seat, prompt, n options, choice, system, who)."""
    n = SB.N_PLAYERS
    res: List[Dict] = [{} for _ in instances]; decisions: List[Tuple] = []; masses: List[float] = []
    reflections: List[Dict] = []; traces: List[List[Dict]] = [[] for _ in instances]
    msgs: List[List[str]] = [[] for _ in instances]; thought: Dict[Tuple[int, int], str] = {}; own = Counter()
    last_ref: Dict[Tuple[int, int], int] = {}; attrib = [_attrib0(n) for _ in instances]
    obs = server.step({i: {"op": "new", "code": x["code"], "n": n, "endowments": x["endowments"], "bystanders": x["bystanders"],
                           "seed": x["seed"], "readouts": True} for i, x in enumerate(instances)})
    live = {}
    for i, ob in obs.items():
        if "error" in ob:
            res[i] = {"error": ob["error"], "detail": ob.get("detail", "")}
        elif ob["over"]:
            res[i] = {"out": ob, "turns": 0, "attrib": attrib[i]}
        else:
            live[i] = ob
    turns = Counter(); bar = tqdm(desc=desc, unit="turn", leave=False, disable=desc is None)
    world = lambda i: (instances[i].get("world", "") + "\n" + roles_text(instances[i]["types"], bool(instances[i].get("public")))).strip()
    system = lambda i, p: latch(instances[i]["types"][p])
    def who_of(i, p):                                       # 'cur' -> 'cur:<seat type>' when each role has its own adapter
        w = instances[i]["who"][p] or "cur"
        return f"cur:{instances[i]['types'][p] or SNAP_ROLE}" if (w == "cur" and nets.roles) else w
    while live:
        ops, llm = {}, []
        for i, ob in live.items():
            seat = instances[i]["seats"][ob["player"]]
            if turns[i] >= SB.MAX_TURNS:
                res[i] = {"error": "too_long_game", "detail": "live"}; continue
            if seat in (NOVICE, RESTRAINED, RAIDER):
                pool = ob["moves"] if seat == NOVICE else (restrained_moves(ob["moves"]) if seat == RESTRAINED else raider_moves(ob["moves"]))
                mv = rng.choice(pool); ops[i] = {"op": "play", "player": ob["player"], "move": mv}
                traces[i].append({"turn": turns[i], "player": ob["player"], "seat": seat, "move": mv, "n_moves": len(ob["moves"])})
            else:
                llm.append(i)
        if llm and think:
            need = [i for i in llm if own[(i, live[i]["player"])] % int(think["every"]) == 0]
            by_who = defaultdict(list)
            for i in need:
                by_who[who_of(i, live[i]["player"])].append(i)
            for who, group in by_who.items():
                texts = [reflect_prompt(instances[i]["rules"], live[i]["describe"], live[i]["moves"], live[i]["player"], world(i), msgs[i]) for i in group]
                ids = [AZ.chat_ids(pol.tok, t, system(i, live[i]["player"])) for t, i in zip(texts, group)]
                with nets.use(who) as m:
                    comps = AZ.generate(m, pol.tok, ids, int(think["max_new"]), float(think.get("temperature", 1.0)), rng.randrange(2 ** 31), int(think["batch"]))
                    miss = [k for k, c in enumerate(comps) if "SAY:" not in pol.tok.decode(c, skip_special_tokens=True)]
                    if miss:                                        # the message never came: ask for it (short, one wave)
                        say_ids = pol.tok("\nSAY:", add_special_tokens=False)["input_ids"]
                        cut = [[t for t in comps[k] if t not in (pol.tok.eos_token_id, pol.tok.pad_token_id)] + say_ids for k in miss]
                        more = AZ.generate(m, pol.tok, [ids[k] + c for k, c in zip(miss, cut)], SAY_FALLBACK_TOKENS,
                                           float(think.get("temperature", 1.0)), rng.randrange(2 ** 31), int(think["batch"]))
                        for k, c, mm in zip(miss, cut, more):
                            comps[k] = c + mm
                for i, pid, cid in zip(group, ids, comps):
                    p = live[i]["player"]; r = parse_reflection(pol.tok.decode(cid, skip_special_tokens=True))
                    thought[(i, p)] = r["think"] + (f"\nI told the others: {r['say']}" if r["say"] else "")
                    if r["say"]:
                        msgs[i].append(f"P{p} (turn {turns[i]}): {r['say']}")
                    reflections.append({"inst": i, "seat": p, "turn": turns[i], "prompt_ids": pid, "comp_ids": cid, "n_moves": 0,
                                        "n_agree": 0, "who": who, "system": system(i, p), **r})
                    last_ref[(i, p)] = len(reflections) - 1
        if llm:
            by_who = defaultdict(list)
            for i in llm:
                by_who[who_of(i, live[i]["player"])].append(i)
            for who, group in by_who.items():
                prompts = [solver_prompt(instances[i]["rules"], live[i]["describe"], live[i]["moves"], live[i]["player"], world(i), msgs[i],
                                         thought.get((i, live[i]["player"]), "")) for i in group]
                systems = [system(i, live[i]["player"]) for i in group]
                with nets.use(who):
                    probs = pol.probs(prompts, [len(live[i]["moves"]) for i in group], systems=systems)
                masses += pol.last_legal_mass
                for i, pr, pt, sy in zip(group, probs, prompts, systems):
                    p = live[i]["player"]; k = PO.sample(pr, gen); decisions.append((i, p, pt, len(pr), k, sy, who)); own[(i, p)] += 1
                    mv = live[i]["moves"][k]; ops[i] = {"op": "play", "player": p, "move": mv}
                    ri = last_ref.get((i, p))
                    if ri is not None:
                        reflections[ri]["n_moves"] += 1; reflections[ri]["n_agree"] += int(acts_on_thought(mv, reflections[ri]["think"]))
                    last = next((r for r in reversed(reflections) if r["inst"] == i and r["seat"] == p and r["turn"] == turns[i]), None) if think else None
                    traces[i].append({"turn": turns[i], "player": p, "seat": LLM, "who": who, "type": instances[i]["types"][p], "move": mv,
                                      "n_moves": len(pr), "p_move": pr[k], "think": last["think"] if last else None,
                                      "say": last["say"] if last else None, "describe": live[i]["describe"][:TRACE_DESCRIBE_CHARS]})
        live = {i: ob for i, ob in live.items() if i in ops}
        if not ops:
            break
        before = {i: (live[i]["res_now"], live[i]["bys_now"]) for i in ops}
        for i, ob in server.step(ops).items():
            turns[i] += 1
            if "error" in ob:
                res[i] = {"error": ob["error"], "detail": ob.get("detail", "")}; live.pop(i, None); continue
            p = ops[i]["player"]; r0, b0 = before[i]; a = attrib[i]; a["moves_by"][p] += 1
            db = [y - x for x, y in zip(b0, ob["bys_now"])]; dr = [y - x for x, y in zip(r0, ob["res_now"])]
            a["bys_by"][p] += sum(max(0, -d) for d in db)
            for j, d in enumerate(dr):
                a["loss_by"][p][j] += max(0, -d)
            traces[i][-1].update(dbys=db, dres=dr)
            if ob["over"]:
                res[i] = {"out": ob, "turns": turns[i], "attrib": a}; live.pop(i, None)
            else:
                live[i] = ob
        bar.update(len(ops))
    bar.close()
    server.step({i: {"op": "drop"} for i in range(len(instances))})
    return res, decisions, masses, reflections, traces


def contest_success(inst: Dict, out: Dict, kind: str) -> float:
    """1 if the seats being tested beat the odd seat out (mean take-out), 0.5 on a tie. ``kind`` 'base' = the base-model
    seats vs a random player; 'past' = the current-model seats vs the past snapshot."""
    r = out["resources"]
    odd = inst["seats"].index(NOVICE) if kind == "base" else inst["who"].index("past")
    m = sum(v for s, v in enumerate(r) if s != odd) / (SB.N_PLAYERS - 1)
    return 1.0 if m > r[odd] else (0.5 if m == r[odd] else 0.0)


def contest_instances(code: str, rules: str, k: int, rng: random.Random) -> List[Tuple[str, Dict]]:
    """The two skill contests for one valid game, ``k`` instances each (all seats latched as type A, types hidden)."""
    out = []
    for j in range(k):
        common = {"code": code, "rules": rules, "endowments": [SB.DEFAULT_ENDOWMENT] * SB.N_PLAYERS,
                  "bystanders": [SB.DEFAULT_BYSTANDER] * SB.N_BYSTANDERS, "types": ["A"] * SB.N_PLAYERS, "public": False,
                  "world": EC.world_text(None)}
        out.append(("base", dict(common, seed=rng.randrange(2 ** 31), seats=novice_seats(j), who=["base"] * SB.N_PLAYERS)))
        who = ["cur"] * SB.N_PLAYERS; who[j % SB.N_PLAYERS] = "past"
        out.append(("past", dict(common, seed=rng.randrange(2 ** 31), seats=[LLM] * SB.N_PLAYERS, who=who)))
    return out


def stop_share(trace: Sequence[Dict], seat: int, stop_verbs: Sequence[str]) -> Optional[float]:
    mv = [m for m in trace if m["player"] == seat]
    return (sum(move_verb(m["move"]) in stop_verbs for m in mv) / len(mv)) if mv else None


# --------------------------------------------------------------------------------------------------------- training
class Buffer3(AZ.Buffer):
    def sample_intervention(self, k: int, rng: random.Random, min_interv: float) -> List[Dict]:
        pool = [it for it in self.items if ((it.get("stats") or {}).get("intervention") or 0.0) >= min_interv] or self.items
        return [pool[rng.randrange(len(pool))] for _ in range(k)]


BASE_CACHE_MAX = 400_000                 # frozen-base scores kept per run (keys are 20-byte digests, values 16 floats)


def _base_key(system: Optional[str], prompt: str, n: int) -> bytes:
    return hashlib.sha1(f"{system}\x00{n}\x00{prompt}".encode()).digest()


def base_logp(pol: PO.LetterPolicy, nets: "Nets", cache: Dict[bytes, torch.Tensor], prompts: Sequence[str], ns: Sequence[int],
              systems: Sequence[Optional[str]], stats: Counter) -> torch.Tensor:
    """The frozen base model's masked letter log-probs [B, 16] for the KL term, scoring only prompts it has not scored before (the
    base never changes, so a score is good for the whole run; 2026-10-04 speed-up)."""
    keys = [_base_key(s, p, n) for p, n, s in zip(prompts, ns, systems)]; miss = [i for i, k in enumerate(keys) if k not in cache]
    stats["base_hit"] += len(keys) - len(miss); stats["base_miss"] += len(miss)
    if miss:
        with torch.no_grad(), nets.use("base"):
            r = pol.forward([prompts[i] for i in miss], [ns[i] for i in miss], systems=[systems[i] for i in miss])["logp"].float().cpu()
        for i, row in zip(miss, r):
            if len(cache) < BASE_CACHE_MAX:
                cache[keys[i]] = row
            else:
                cache.setdefault(keys[i], row)
    rows = [cache.get(k) for k in keys]
    if any(r is None for r in rows):                       # cache full and evicted nothing: score the rest directly
        with torch.no_grad(), nets.use("base"):
            r = pol.forward(list(prompts), list(ns), systems=list(systems))["logp"].float().cpu()
        rows = [x if x is not None else r[i] for i, x in enumerate(rows)]
    return torch.stack(rows)


def make_worlds(comps: Sequence[str], per_comp: int, seed: int) -> List[Dict]:
    """Persistent worlds, ``per_comp`` per composition; each fixes a seat order of its types for its whole life."""
    rng = random.Random(seed * 31 + 7); out = []
    for c in comps:
        if c not in COMPOSITIONS:
            raise ValueError(f"composition {c!r} (valid: {COMPOSITIONS})")
        for k in range(per_comp):
            types = list(c); rng.shuffle(types)
            out.append({"comp": c, "types": types, "w": EC.World(seed=seed * 1000 + len(out))})
    return out


def train_run3(nets: Nets, tok, pol: PO.LetterPolicy, seeds: Sequence[Dict], cfg: Dict, spec: Dict, out_dir: Path, n_steps: int,
               max_minutes: Optional[float], pool: SB.Pool, server: SB.GameServer, inject: Optional[str] = None) -> Dict:
    """One night-3 run: proposer + type A + type B in one set of weights. ``spec``: name, seed, types {A: rule, B: rule},
    compositions (default all four)."""
    seed = int(spec["seed"]); rules_of_type = dict(spec.get("types") or {"A": "finite", "B": "infinite"})
    comps = list(spec.get("compositions") or COMPOSITIONS)
    restore_prob, bitter_games = float(cfg["stomp_restore_prob"]), int(cfg["bitter_games"])
    stomp_mode = spec.get("stomp_mode") or cfg.get("stomp_mode", "restart")         # 2026-10-03 regimes: "persist" = stomps restart nothing
    if stomp_mode == "persist":
        bitter_games = 0
    out_dir.mkdir(parents=True, exist_ok=True); rng = random.Random(seed); buf = Buffer3(seeds, cfg.get("max_ref_chars"))
    worlds = make_worlds(comps, int(cfg["worlds_per_comp"]), seed); E_ = int(cfg["economy_games"])
    params = nets.trainable(); lr = float(cfg["lr_full"] if nets.mode == "full" else cfg["lr"])
    groups = nets.role_params()
    if nets.roles:                           # the proposer adapter gets its own learning rate (night 4: RL never moved it)
        opt = torch.optim.AdamW([{"params": ps, "lr": float(cfg.get("lr_proposer", lr)) if r == "P" else lr} for r, ps in groups.items()], weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    P, G = int(cfg["proposals"]), int(cfg["group"]); t0 = time.time(); done = 0; stopped = "steps"; valid_hist: List[float] = []
    think = cfg.get("think") if cfg.get("think", {}).get("enabled", True) else None
    tel = open(out_dir / "train_telemetry.jsonl", "a"); props = open(out_dir / "proposals.jsonl", "a"); trf = open(out_dir / "traces.jsonl", "a")
    bar = tqdm(total=n_steps, desc=f"azr3 {spec['name']}", unit="step"); snaps = out_dir / "snapshots"
    base_cache: Dict[bytes, torch.Tensor] = {}; save_every = int(cfg.get("save_every") or 0)
    for step in range(n_steps):
        if max_minutes is not None and (time.time() - t0) / SECONDS_PER_MINUTE > float(max_minutes):
            stopped = "time"; break
        ts = time.time(); gen = torch.Generator().manual_seed(seed * 1_000_003 + step); gate = EC.quality_gate(valid_hist)
        tim: Counter = Counter(); cnt: Counter = Counter()      # seconds per phase and base-cache hits (2026-10-04: where a step's time goes)
        # 1-2: propose and validate
        refs = [[r["code"] for r in buf.sample(int(cfg["n_refs"]), rng, refs=True)] for _ in range(P)]
        prompts = [AZ.chat_ids(tok, proposer_prompt3(rf), PROPOSER_SYSTEM3) for rf in refs]
        comps_ids = []
        if P:                                # proposals: 0 = the players train on the seed games alone (sanction night)
            with nets.use("cur:P") as m:
                comps_ids = AZ.generate(m, tok, prompts, int(cfg["max_new_tokens"]), float(cfg["temperature"]), seed * 7919 + step, int(cfg["gen_batch"]))
        texts = [tok.decode(c, skip_special_tokens=True) for c in comps_ids]
        codes = [AZ.extract_code(t) for t in texts]; verdicts: List[Optional[Dict]] = [None] * P
        if inject and step == 0:
            codes[0] = inject
        known = {AZ._norm(it["code"]) for it in buf.items}
        for i, c in enumerate(codes):
            if c is not None and (not inject or step or i) and AZ._norm(c) in known:
                verdicts[i] = {"ok": False, "reason": "copy", "detail": "a game already in the buffer, verbatim"}
        todo = [i for i, c in enumerate(codes) if c is not None and verdicts[i] is None]
        for i, v in zip(todo, pool.validate_many([codes[i] for i in todo], seed=step, readouts=True, affordance=True) if todo else []):
            verdicts[i] = v
        for i in range(P):
            if verdicts[i] is None:
                verdicts[i] = {"ok": False, "reason": "no_code", "detail": ""}
        valid = [i for i in range(P) if verdicts[i]["ok"]]
        # 3: contests for every valid game + economy games, in one lockstep batch
        inst: List[Dict] = []; meta: List[Tuple] = []
        for i in valid:
            for kind, x in contest_instances(codes[i], verdicts[i]["stats"]["rules"], int(cfg["learn_instances"]), rng):
                inst.append(x); meta.append(("contest", i, kind))
        games = []
        props_left = [(codes[i], verdicts[i]["stats"]["rules"], "proposal", verdicts[i]["stats"]) for i in valid]
        slots = [worlds[(step * E_ + j) % len(worlds)] for j in range(E_)]
        for j, wd in enumerate(slots):
            if props_left:
                games.append(props_left.pop(0))
            else:
                mixed = len(set(wd["types"])) > 1
                it = (buf.sample_intervention(1, rng, float(cfg["intervention_min"]))[0] if mixed and rng.random() < float(cfg["intervention_bias"])
                      else buf.sample(1, rng)[0])
                games.append((it["code"], it["rules"], it.get("source", "buffer"), it.get("stats") or {}))
        before = []
        for j, (code, rules, src, st) in enumerate(games):
            wd = slots[j]; w = wd["w"]; endow = w.endowments(); before.append((wd, endow, EC.World(**vars(w))))
            gs = rng.randrange(2 ** 31); public = rng.random() < float(cfg["public_prob"])
            for r in range(G):
                inst.append({"code": code, "rules": rules, "endowments": list(endow), "bystanders": list(w.bystanders), "seed": gs + r,
                             "seats": [LLM] * SB.N_PLAYERS, "types": list(wd["types"]), "who": ["cur"] * SB.N_PLAYERS, "public": public,
                             "world": EC.world_text(w)})
                meta.append(("econ", j, r))
        tim["propose"] = time.time() - ts; t_ = time.time()
        results, decs, masses, refl, traces = play_batch3(server, pol, nets, inst, rng, gen, think=think) if inst else ([], [], [], [], [])
        tim["play"] = time.time() - t_; t_ = time.time()
        # 4: rewards
        succ: Dict[Tuple[int, str], List[float]] = defaultdict(list); broken = set()
        for (kind, a, b), x, res in zip(meta, inst, results):
            if kind == "contest":
                if "error" in res:
                    broken.add(a)
                else:
                    succ[(a, b)].append(contest_success(x, res["out"], b))
        p_rew, parts = [], []
        for i in range(P):
            if i in broken:
                verdicts[i] = {"ok": False, "reason": "runtime_in_play", "detail": "", "stats": verdicts[i].get("stats")}
            rates = {k: sum(succ[(i, k)]) / len(succ[(i, k)]) for k in ("base", "past") if succ.get((i, k))}
            sr = max(rates.values(), key=EC.learnability) if rates else None      # the contest with the better learnability
            rw, pt = EC.proposer_reward3(verdicts[i], sr, gate, len(codes[i] or texts[i]))
            p_rew.append(rw); parts.append(pt); verdicts[i]["success_rate"] = sr if verdicts[i]["ok"] else None; verdicts[i]["contests"] = rates
        s_rew: Dict[Tuple[int, int], List[float]] = {}; econ = []; pos = {mm: k for k, mm in enumerate(meta)}
        credit = float(spec.get("sanction_credit") or 0.0); credits: List[float] = []
        for j in range(len(games)):
            wd, endow, snap = before[j]; outs = {b: res for (kind, a, b), res in zip(meta, results) if kind == "econ" and a == j}
            if any("error" in o for o in outs.values()) or len(outs) != G:
                continue
            rules = [rules_of_type[t] for t in wd["types"]]
            for r in range(G):
                ev = EC.game_outcome(snap, outs[r]["out"], bitter_games); s_rew[(j, r)] = EC.seat_rewards(rules, endow, ev, outs[r]["attrib"])
                if credit:                   # the credit run: B seats also earn a bonus per JUSTIFIED sanction (azr_sanction)
                    cr = SN.sanction_credit(traces[pos[("econ", j, r)]], SN.player_verbs(games[j][1]), SB.N_PLAYERS, credit,
                                            float(cfg["sanction_credit_cap"]), SN.rule_of(games[j][1]))
                    for s_, rule in enumerate(rules):
                        if rule.startswith("infinite"):
                            s_rew[(j, r)][s_] += cr[s_]; credits.append(cr[s_])
            ev0 = EC.settle(wd["w"], endow, outs[0]["out"], restore_prob=restore_prob, bitter_games=bitter_games, stomp_mode=stomp_mode)
            at = outs[0]["attrib"]; bys_in = max(1, sum(snap.bystanders))
            econ.append(dict(ev0, comp=wd["comp"], types=wd["types"], source=games[j][2],
                             take_by_type={t: sum(at["bys_by"][s] for s, tt in enumerate(wd["types"]) if tt == t) / bys_in for t in set(wd["types"])}))
        # 5: update (proposer + the two player types; only the current weights' moves in economy games are trained)
        opt.zero_grad(set_to_none=True); base_p = sum(p_rew) / max(1, P); kl_p = kl_s = kl_t = 0.0; n_p = 0
        nets.model.train()
        for i in range(P):
            adv = p_rew[i] - base_p
            if abs(adv) < 1e-9 or not comps_ids[i]:
                continue
            with torch.no_grad(), nets.use("base") as m:
                lr_ = AZ.batch_completion_logprob(m, [prompts[i]], [comps_ids[i]], tok.pad_token_id)[0]
            with nets.use("cur:P") as m:
                lp = AZ.batch_completion_logprob(m, [prompts[i]], [comps_ids[i]], tok.pad_token_id)[0]
                d = lr_ - lp; kl = (torch.exp(d) - d - 1).mean()
                ((-adv * lp.mean() + float(cfg["beta_proposer"]) * kl) / P).backward()
            kl_p += float(kl); n_p += 1
        idx = {mm: k for k, mm in enumerate(meta)}; grp = {}
        for j in range(len(games)):
            if (j, 0) not in s_rew:
                continue
            for seat in range(SB.N_PLAYERS):
                vals = [s_rew[(j, r)][seat] for r in range(G)]; mu = sum(vals) / G
                for r in range(G):
                    grp[(idx[("econ", j, r)], seat)] = vals[r] - mu
        sd = sorted([d for d in decs if (d[0], d[1]) in grp and d[6].startswith("cur")], key=lambda d: d[6]); n_all = max(1, len(sd))
        kl_role: Dict[str, List[float]] = defaultdict(list)
        by_role = [[d for d in sd if d[6] == w] for w in sorted({d[6] for d in sd})]
        mbs = [part[s0:s0 + pol.micro_batch] for part in by_role for s0 in range(0, len(part), pol.micro_batch)]   # one role per micro-batch
        for mb in mbs:
            pr_, n_, sy_ = [d[2] for d in mb], [d[3] for d in mb], [d[5] for d in mb]; who_mb = mb[0][6]
            t_b = time.time(); refp = {"logp": base_logp(pol, nets, base_cache, pr_, n_, sy_, cnt)}; tim["base_ref"] += time.time() - t_b
            with nets.use(who_mb):
                cur = pol.forward(pr_, n_, systems=sy_)
                ch = torch.tensor([d[4] for d in mb], device=cur["logp"].device)
                a = torch.tensor([grp[(d[0], d[1])] for d in mb], device=cur["logp"].device, dtype=cur["logp"].dtype)
                ke = PO.kl_and_entropy(cur["logp"], refp["logp"].to(cur["logp"].device, cur["logp"].dtype))
                ((-(a * cur["logp"].gather(1, ch[:, None]).squeeze(1)) + float(cfg["beta_solver"]) * ke["kl"]).sum() / n_all).backward()
            kl_s += float(ke["kl"].sum()); kl_role[who_mb] += ke["kl"].detach().float().cpu().tolist()
        tim["update_moves"] = time.time() - t_; t_ = time.time()
        tr = sorted([r for r in refl if (r["inst"], r["seat"]) in grp and r["comp_ids"] and r["who"].startswith("cur")], key=lambda r: r["who"])
        cons = [r["n_agree"] / r["n_moves"] if r["n_moves"] else 0.0 for r in tr]; mc = sum(cons) / len(cons) if cons else 0.0
        tw, cw = float(cfg["think"].get("loss_weight", 1.0)), float(cfg["think"].get("consistency_weight", 0.0)); tb = int(cfg["think"]["train_batch"])
        for r, c in zip(tr, cons):
            r["adv"] = grp[(r["inst"], r["seat"])] + cw * (c - mc)
        tmbs = [part[s0:s0 + tb] for part in [[r for r in tr if r["who"] == w] for w in sorted({r["who"] for r in tr})] for s0 in range(0, len(part), tb)]
        for mb in tmbs:
            P_, C_ = [r["prompt_ids"] for r in mb], [r["comp_ids"] for r in mb]
            with torch.no_grad(), nets.use("base") as m:
                lrs = AZ.batch_completion_logprob(m, P_, C_, tok.pad_token_id)
            with nets.use(mb[0]["who"]) as m:
                lps = AZ.batch_completion_logprob(m, P_, C_, tok.pad_token_id); loss = 0.0
                for r, lp, lr_ in zip(mb, lps, lrs):
                    d = lr_ - lp; kl = (torch.exp(d) - d - 1).mean(); kl_t += float(kl)
                    loss = loss + tw * (-r["adv"] * lp.mean() + float(cfg["beta_solver"]) * kl) / len(tr)
                loss.backward()
        tim["update_think"] = time.time() - t_
        gn_role = {}                         # per-role gradient norm BEFORE clipping; separate adapters are clipped separately
        for r, ps in groups.items():
            gs = [q.grad.detach().float().norm() for q in ps if q.grad is not None]
            gn_role[r] = float(torch.norm(torch.stack(gs))) if gs else 0.0
            if nets.roles and gs:
                torch.nn.utils.clip_grad_norm_([q for q in ps if q.grad is not None], float(cfg["grad_clip"]))
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]) * (len(groups) ** 0.5 if nets.roles else 1.0)))
        opt.step(); nets.model.eval(); done = step + 1
        if save_every and done % save_every == 0 and nets.roles:      # adapter checkpoints along a long run, for later evaluation
            nets.model.save_pretrained(str(out_dir / "checkpoints" / f"step_{done}"), selected_adapters=list(nets.roles.values()))
        if int(cfg["snap_every"]) and (step + 1) % int(cfg["snap_every"]) == 0:
            nets.snapshot(step, snaps, int(cfg["snap_ring_full"] if nets.mode == "full" else cfg["snap_ring"]))
        valid_hist.append(len(valid) / max(1, P))
        # 6: logs
        for j in range(len(games)):
            k = idx.get(("econ", j, 0))
            if k is None or "error" in results[k]:
                continue
            o = results[k]["out"]; wd, endow, snap = before[j]
            trf.write(json.dumps({"step": step, "game": j, "source": games[j][2], "comp": wd["comp"], "types": wd["types"],
                                  "public": inst[k]["public"], "world_before": {"wallets": snap.wallets, "bystanders": snap.bystanders,
                                  "epoch_games": snap.epoch_games}, "rules": games[j][1][:1500], "moves": traces[k], "resources": o["resources"],
                                  "bystanders_after": o["bystanders"], "attrib": results[k]["attrib"], "rewards": s_rew.get((j, 0))}) + "\n")
        trf.flush()
        for i in range(P):
            st = verdicts[i].get("stats") or {}
            rec = {"step": step, "i": i, "reason": None if verdicts[i]["ok"] else verdicts[i]["reason"], "detail": verdicts[i].get("detail", "")[:300],
                   "reward": p_rew[i], "gate": gate, "parts": parts[i], "contests": verdicts[i].get("contests"), "success_rate": verdicts[i].get("success_rate"),
                   "chars": len(codes[i] or ""), "code": codes[i], "rules": st.get("rules") or rules_of(codes[i]),
                   "text": texts[i] if codes[i] is None else None, "stats": {k: v for k, v in st.items() if k != "rules"}}
            props.write(json.dumps(rec) + "\n")
            if verdicts[i]["ok"]:
                buf.add({"code": codes[i], "rules": st["rules"], "source": f"s{seed}_step{step}", "stats": rec["stats"]})
        props.flush()
        vparts = [parts[i] for i in valid]
        sanc: Dict[str, List[float]] = defaultdict(list)       # replay 0 of every economy game: sanctions per seat type and game kind
        for j in range(len(games)):
            k = pos.get(("econ", j, 0))
            if k is None or "error" in results[k]:
                continue
            verbs = SN.player_verbs(games[j][1]); kind = "ctl" if SN.is_control_game(games[j][1]) else "eff"
            if not verbs:
                continue
            for s_, t in enumerate(before[j][0]["types"]):
                st = SN.seat_stats(traces[k], verbs, before[j][0]["types"], s_, SN.rule_of(games[j][1]))
                if st["sanction_share"] is not None:
                    sanc[f"sanction_share_{t}_{kind}"].append(st["sanction_share"]); sanc[f"sanction_just_{t}_{kind}"].append(st["sanction_just"])
        row = {"step": step, "run": spec["name"], "seed": seed, "seconds": round(time.time() - ts, 1), "gate": gate,
               "valid_share": len(valid) / max(1, P), "reasons": dict(Counter(v["reason"] for v in verdicts if not v["ok"])),
               "proposer_reward": base_p, "quality": AZ._mean([p.get("quality") for p in vparts]),
               **{f"q_{k}": AZ._mean([p.get(k) for p in vparts]) for k in EC.QUALITY_WEIGHTS},
               "success_rate": AZ._mean([verdicts[i].get("success_rate") for i in valid]), "chars": AZ._mean([len(c or "") for c in codes]),
               "buffer": len(buf.items), "economy_games": len(econ), "past_step": nets.past_step,
               "restarts": sum(e["restarted"] for e in econ), "ruins": sum(e["ruin"] for e in econ), "stomps": sum(e["stomp"] for e in econ),
               "bystander_loss": AZ._mean([e["bystander_loss"] for e in econ]),
               **{f"bystander_loss_{c}": AZ._mean([e["bystander_loss"] for e in econ if e["comp"] == c]) for c in comps},
               **{f"take_{t}_{c}": AZ._mean([e["take_by_type"].get(t) for e in econ if e["comp"] == c]) for c in comps for t in "AB" if t in c},
               "solver_reward_A": AZ._mean([s_rew[(j, r)][s] for (j, r) in s_rew for s in range(SB.N_PLAYERS) if before[j][0]["types"][s] == "A"]),
               "solver_reward_B": AZ._mean([s_rew[(j, r)][s] for (j, r) in s_rew for s in range(SB.N_PLAYERS) if before[j][0]["types"][s] == "B"]),
               "solver_decisions": len(sd), "llm_decisions": len(decs), "kl_proposer": kl_p / max(1, n_p), "kl_solver": kl_s / n_all,
               "kl_think": kl_t / max(1, len(tr)), "grad_norm": gnorm,
               **{f"grad_norm_{r}": v for r, v in gn_role.items()}, **{f"kl_solver_{w.split(':')[-1]}": AZ._mean(v) for w, v in kl_role.items()}, "reflections": len(refl), "reflections_trained": len(tr),
               "messages_sent": sum(bool(r["say"]) for r in refl), "think_consistency": AZ._mean([r["n_agree"] / r["n_moves"] for r in refl if r["n_moves"]]),
               "legal_mass": AZ._mean(masses), "gen_tokens": AZ._mean([len(c) for c in comps_ids]),
               "world_restarts_total": sum(wd["w"].restarts for wd in worlds),
               **{k: AZ._mean(v) for k, v in sanc.items()}, "sanction_credit": AZ._mean(credits) if credit else None,
               **{f"t_{k}": round(v, 2) for k, v in tim.items()}, "base_cache_hit": cnt["base_hit"], "base_cache_miss": cnt["base_miss"],
               "base_cache_size": len(base_cache), "infer_batch": pol.infer_batch, "micro_batch": pol.micro_batch}
        tel.write(json.dumps(row) + "\n"); tel.flush()
        bar.update(1); bar.set_postfix(valid=f"{row['valid_share']:.2f}", gate=f"{gate:.2f}", pr=f"{base_p:.2f}", rst=row["restarts"])
    bar.close(); tel.close(); props.close(); trf.close()
    with open(out_dir / "buffer.jsonl", "w") as f:
        for it in buf.items:
            f.write(json.dumps({k: it[k] for k in ("code", "rules", "source", "stats") if k in it}) + "\n")
    with open(out_dir / "worlds.json", "w") as f:
        json.dump([{"comp": wd["comp"], "types": wd["types"], "world": vars(wd["w"])} for wd in worlds], f)
    return {"steps_done": done, "stopped": stopped, "minutes": (time.time() - t0) / SECONDS_PER_MINUTE, "buffer": len(buf.items),
            "past_step": nets.past_step}
