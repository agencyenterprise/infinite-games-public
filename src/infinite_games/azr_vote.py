"""The voting version of the infinite game (Thomas 2026-10-06): a proposer and finite players, with the infinite part supplied by
votes, exit and the fun terms instead of hand-coded world penalties. Rules and rewards: vote_rules.py.

One training update:
    1. propose     the proposer adapter writes ``proposals`` games; the sandbox validates them; valid ones join the buffer and are
                   offered first. The seed buffer holds what earlier nights learned (the hand-written games, the sanction games with
                   full return and guilty pays). Proposer reward: validity (+1, partial credit by failing stage, length penalty) plus
                   VOTE_BONUS x what the voters thought of its game when it was offered this update (voting runs only).
    2. offer       ``offerings`` games per update, each in its own persistent world (wallets carry; a stomp restarts nothing and
                   bystanders regrow, stomp_mode persist). Every ``anchor_every`` updates one offering is a dull anchor (no choice
                   matters), so a vote that accepts dull games can be seen.
    3. per replay (``group`` replays of every offering, different seeds):
         vote 1     each player votes play / decline / reset-then-play (reset only while the world has resets left); in the
                    bystander condition each bystander votes allow / refuse. Majority of voters: the game is played; a decliner's
                    seat goes to a passive stand-in. azr21 (vote.lottery_share): played with probability = the weighted play share,
                    the bystanders together holding lottery_share of the weight (vote_rules.lottery).
         game 1     played; each player seat scored by gain + fun terms (vote_rules.seat_rewards).
         vote 2     the same game is offered again (the rematch) and voted the same way.
         game 2     played if the rematch passes.
         Pyrrhic    if most of the players who played game 1 decline the rematch, game-1 winners' positive gains turn negative.
       Votes are moves: a play vote earns the game's reward (0 if the game did not go ahead), a decline earns the decline reward.
    4. update      every LLM decision (votes and moves) is reinforced by its return minus the mean over the replays of the same
                   offering, with the KL penalty to the frozen base; proposer and player adapters (and the bystander adapter in the
                   bystander condition) have their own optimiser groups and clips.
The CONTROL run (spec vote.enabled false): no votes, no fun terms, no Pyrrhic rule; every game and rematch is played; finite gains.

Knobs added 2026-10-07 for the azr9 grid (spec ``vote``; absent = the night-8 rules):
    decline_reward / decline_decay   size of a player's decline reward and its halving per decline in a row
    centre_fun                       fun terms enter the reward minus the running baseline of the weighted fun total (vote_rules)
    plant_growth                     bystanders are plants: an allowed, played game pays growth + the change in holdings
    bystander_decline_reward         what a bystander's refusal pays (default 0 for plants, else the players' decline reward)
    harm_veto / veto_penalty         (2026-10-10, azr10) a bystander harmed in game 1 vetoes the rematch alone; players who took from a
                                     vetoing bystander lose veto_penalty from their game-1 return (vote_rules.harm_veto / veto_penalties)
    veto_voter_penalty               and every player who voted for the vetoed rematch loses this on that vote (rematch_voter_penalties)
cfg ``require_effect`` (azr10): proposals are valid only if holdings change in the sandbox's random plays (azr_sandbox.validate).
    control: {gains: pid | integral | {kp, ki, kd}}   (2026-10-12, azr11) PID-Lagrangian controllers (vote_control.py) steer the decline
                                     share, the harmful-game share and the bystanders' allow share into target bands by rescaling the decline
                                     reward / cost, the group punishments (Pyrrhic strength, voter penalty), the takers' penalty and plant growth
    table_types: [[A, A, W], ...]    (2026-10-14, azr12) seat types per world (cycled over the worlds; absent = all A). W = PURE WINNERS with
                                     their own adapter: reward = finite gain + their own fun terms (no 'others' term, own centred baseline);
                                     exempt from every social rule (Pyrrhic flip, takers' veto penalty, voter penalty; no controller decline
                                     bonus / cost). Others can still decline, veto or sanction them in-game. The controllers read only A seats:
                                     decline share of A voters, harmful share = played game 1s in which an A seat took from a bystander.
                                     Every vote prompt states the table's types (also at all-A tables, so the line itself is no cue); at
                                     tables with a winner, vote and game prompts add WINNER_NOTE (winners never pay the rematch rules).
cfg ``predict`` (azr15, Thomas 2026-10-19; vote_predict.py): every new valid game is first played once in a scratch copy of its world
                                     (trial: not trained, changes nothing), and every seat remembers what it last saw of a game (trial or real
                                     game 1); before the first vote each seat predicts its reward PART BY PART (gain, social rules, each fun
                                     term) and then the total from the shown mix and its own part predictions, as bands of running quintiles;
                                     the vote prompt carries the memory and the predicted total band. Predictions are trained SUPERVISED on the
                                     bands the real game 1 lands in; the proposer's bonus is the type-A seats' expected predicted total.
cfg ``recurring`` {share, recent_n} (azr15): a buffer offering is drawn with probability ``share`` from the ``recent_n`` most recently
                                     added proposer games, so games come back and the experience-based fun terms can move.
vote.control.fun_share_lo (azr15): a PID constraint holds the type-A fun share >= this by scaling the A fun weights (vote_control).
"""
from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from . import azr as AZ
from . import azr3 as A3
from . import azr_economy as EC
from . import azr_sanction as SN
from . import azr_sandbox as SB
from . import engine as E
from . import fun_select as FS
from . import policy as PO
from . import proposer_bench as PB
from . import vote_control as VC
from . import vote_predict as VP
from . import vote_rules as VR
from .azr_text import LLM, PROPOSER_SYSTEM3, RESTRAINED, latch, roles_text

SECONDS_PER_MINUTE = 60.0
PLAYER, BYSTANDER = "A", "B"             # role adapters: players train as role A; voting bystanders as role B
REF_ROLE = "R"                            # azr14: frozen copy of the warm-started proposer (KL anchor)
PRED_ROLE = "Q"                           # azr15: the self-prediction adapter
WINNER = "W"                              # azr12: pure-winner players (own adapter, finite reward, no social penalties)
BYSTANDER_SYSTEM = "You are a bystander in this world: you hold resources, you never make moves in the games, and the players may take from you."
VOTE_BONUS = 1.0                          # proposer: weight on the voters' verdict on its game
PLAYER_OPTIONS = {VR.PLAY: "play", VR.DECLINE: "decline (you sit this game out; a passive stand-in takes your seat)",
                  VR.RESET: "reset: vote to set every player's resources back to the start, then play"}
BYSTANDER_OPTIONS = {VR.PLAY: "allow this game to be played", VR.DECLINE: "refuse this game"}
VETO_WORLD = (" A bystander who loses holdings in a game can veto its rematch on its own; then every player who took from it in that game"
              " is penalised, and so is every player who voted for the rematch.")
VETO_NOTE = "You lost holdings in the last game, so your refusal alone blocks this rematch."
WINNER_NOTE = " Type W players play only for their own winnings; the rematch rules (vetoes, the Pyrrhic rule, the penalties) never cost them anything."
TRACE_CHARS = 1200


def world_text(w: EC.World, veto: bool = False) -> str:
    """The persist-world counterpart of azr_economy.world_text (which describes the restart world)."""
    return (f"This game is one of a continuing series played by the same three players (P0, P1, P2) and the same two bystanders (B0, B1). "
            f"Game {w.games + 1} of this world. What each player takes out carries into the next game (each brings in up to {EC.ENTRY_CAP}). "
            "Wallets now: " + ", ".join(f"P{i} {x}" for i, x in enumerate(w.wallets)) + ". Bystanders' holdings now: "
            + ", ".join(f"B{i} {b}" for i, b in enumerate(w.bystanders)) + f". If a player ends a game with less than {EC.MIN_ENTRY} "
            f"the world restarts for everyone. A bystander's losses are not restored; it regrows {EC.BYSTANDER_REGROW} per game."
            + (VETO_WORLD if veto else ""))


def vote_prompt(rules: str, w: EC.World, seat: int, bystander: bool, rematch: bool, last: Optional[str], options: Sequence[str],
                veto: bool = False, note: Optional[str] = None, types: Optional[Sequence[str]] = None, extra: Optional[str] = None) -> str:
    who = (f"You are bystander B{seat}. You hold {w.bystanders[seat]}." if bystander else f"You are P{seat}.") + (f" {note}" if note else "")
    head = "The same game is offered again: a rematch of the game just played." if rematch else "A game is offered."
    last_txt = f"\nHow the last game went: {last}" if last else ""
    opts = "\n".join(f"{E.LETTERS[i]}) {o}" for i, o in enumerate(options))
    q = "Do you allow this game?" if bystander else "Do you want to play it?"
    tt = (f" {roles_text(types, True)}" + (WINNER_NOTE if WINNER in types else "")) if types else ""
    ex = f"\n\n{extra}" if extra else ""
    return f"{world_text(w, veto)}{tt}\n\n{head} Its rules:\n{rules}{last_txt}{ex}\n\n{who} {q}\n\nYour options:\n{opts}\n\nReply with the letter of your choice."


def game_head(rules: str, w: EC.World, seat: int, types: Optional[Sequence[str]], veto: bool, memory: Optional[str]) -> str:
    """azr15: the context of a prediction prompt (the first-offer vote prompt without the question)."""
    tt = (f" {roles_text(types, True)}" + (WINNER_NOTE if WINNER in types else "")) if types else ""
    return f"{world_text(w, veto)}{tt}\n\nA game is offered. Its rules:\n{rules}" + (f"\n\n{memory}" if memory else "") + f"\n\nYou are P{seat}."


def _fits(pol: PO.LetterPolicy, prompt: str, system: Optional[str]) -> bool:
    try:
        pol.encode(prompt, system); return True
    except ValueError:
        return False


def _resets_left(wd: Dict) -> int:
    return VR.RESETS_PER_EPOCH - sum(1 for g in wd["resets"] if wd["w"].games - g < VR.RESET_EPOCH)


def _summary(out: Dict, endow: Sequence[int], bys_in: Sequence[int]) -> str:
    return (f"players went from {list(endow)} to {list(out['resources'])}; bystanders from {list(bys_in)} to {list(out['bystanders'])}.")


def train_vote_run(nets: A3.Nets, tok, pol: PO.LetterPolicy, seeds: Sequence[Dict], anchors: Sequence[Dict], cfg: Dict, spec: Dict,
                   out_dir: Path, n_steps: int, max_minutes: Optional[float], pool: SB.Pool, server: SB.GameServer) -> Dict:
    vc = dict(spec.get("vote") or {}); voting = bool(vc.get("enabled", True)); bys_vote = voting and bool(vc.get("bystanders_vote"))
    fun = voting and bool(vc.get("fun", True)); pyr = voting and bool(vc.get("pyrrhic", True))
    d_base = float(vc.get("decline_reward", VR.DECLINE_REWARD)); d_decay = float(vc.get("decline_decay", VR.DECLINE_DECAY))
    centre = fun and bool(vc.get("centre_fun", False)); growth = vc.get("plant_growth")
    b_base = float(vc.get("bystander_decline_reward", 0.0 if growth is not None else d_base))
    veto = bys_vote and bool(vc.get("harm_veto", False)); veto_pen = float(vc.get("veto_penalty", VR.VETO_PENALTY))
    b_seats = int(vc.get("bystander_seats", 1))                          # azr20: voting seats per bystander (2 -> 4 bystander votes vs 3 players)
    if b_seats < 1 or (b_seats > 1 and (veto or not bys_vote)):
        raise ValueError(f"vote.bystander_seats {b_seats}: >= 1, and > 1 only with bystanders_vote on and harm_veto off")
    n_bv = SB.N_BYSTANDERS * b_seats if bys_vote else 0                  # bystander voters; voter s >= N_PLAYERS speaks for bystander (s - N_PLAYERS) // b_seats
    b_share = bool(vc.get("bystander_share", False))                     # azr20: bystander loss as a share of its own holdings
    # azr21: LOTTERY vote. The game is played with probability = the weighted play share; the bystanders together hold lottery_share of
    # the vote weight (each player weighs 1). lottery_share 0 = the players-only lottery (bystanders_vote off: the 0% control).
    lot = vc.get("lottery_share"); w_lot = 0.0
    if lot is not None:
        lot = float(lot)
        if not voting or veto or b_seats != 1 or (lot == 0.0) == bys_vote:
            raise ValueError(f"vote.lottery_share {lot}: needs voting on, harm_veto off, bystander_seats 1, and bystanders_vote on iff the share > 0")
        w_lot = VR.bystander_vote_weight(lot, SB.N_PLAYERS, SB.N_BYSTANDERS) if lot > 0 else 0.0
    veto_vpen = float(vc.get("veto_voter_penalty", VR.VETO_VOTER_PENALTY))
    require_effect = bool(cfg.get("require_effect", False))
    pc = dict(cfg.get("proposer") or {})                                # azr13: proposer settings (absent = the azr12 behaviour)
    p_temp = float(pc.get("temperature", cfg["temperature"])); p_variant = pc.get("prompt", "current")
    p_fix = bool(pc.get("fix_imports", False)); p_repairs = int(pc.get("repairs", 0))
    vote_all = bool(pc.get("vote_all", False)); p_degen = bool(pc.get("degenerate_guard", False)); rb = pc.get("rollback")   # azr14
    fun_v2 = fun and bool(vc.get("fun_v2", False))                       # azr14: agency + outcome-uncertainty fun terms
    w_A = VR.FUN_WEIGHTS_V2 if fun_v2 else None; w_W = VR.WINNER_FUN_WEIGHTS_V2 if fun_v2 else VR.WINNER_FUN_WEIGHTS
    w_base = {PLAYER: w_A if w_A is not None else VR.FUN_WEIGHTS, WINNER: w_W}      # azr19: fun scores use the unscaled weights
    table_types = [list(t) for t in (vc.get("table_types") or [])]      # azr12
    if any(len(t) != SB.N_PLAYERS or any(x not in (PLAYER, WINNER) for x in t) for t in table_types):
        raise ValueError(f"vote.table_types {table_types}: each table is {SB.N_PLAYERS} of {PLAYER!r} / {WINNER!r}")
    # azr17: the FOCUS type is the player type whose seats the decline controller, the fun mixer, the proposer's bonus and the vote probe read
    # (default A = every run before azr17; W for the finite-only run, which has no A seats). Mixed tables keep A.
    focus = vc.get("focus", PLAYER)
    if focus not in (PLAYER, WINNER) or (table_types and not any(focus in t for t in table_types)):
        raise ValueError(f"vote.focus {focus!r}: one of {PLAYER!r} / {WINNER!r}, and present at some table")
    ag_key = {"old": "agency", "strict": "agency_strict"}.get(vc.get("fun_agency", "old"))   # azr17: which sandbox agency feeds the fun term
    if ag_key is None:
        raise ValueError(f"vote.fun_agency {vc.get('fun_agency')!r}: old or strict")
    cc = vc.get("control") if voting else None                          # azr11: dynamic reward weights
    gains = None if not cc else ({"pid": VC.PID, "integral": VC.INTEGRAL_ONLY}.get(cc.get("gains", "pid")) if isinstance(cc.get("gains", "pid"), str) else cc["gains"])
    if cc and gains is None:
        raise ValueError(f"vote.control.gains {cc.get('gains')!r}: use pid, integral or {{kp, ki, kd}}")
    ctl = VC.Controllers(dict(gains), ema=float(cc.get("ema", VC.EMA)), fun_share_lo=cc.get("fun_share_lo"),   # azr16: ema, max_step
                         max_step=cc.get("max_step"), harm=bool(cc.get("harm", True))) if cc else None   # azr18: harm False
    prc = dict(cfg.get("predict") or {}); predict = voting and bool(prc.get("enabled", False))     # azr15
    trial = predict and bool(prc.get("trial", True)); use_mem = predict and bool(prc.get("memory", True)); pred_w = float(prc.get("weight", 1.0))
    mem_others = use_mem and bool(prc.get("memory_others", False))      # azr19: the memory note also gives the other players' holdings
    binner = VP.Binner(int(prc.get("window", VP.WINDOW)), int(prc.get("min_n", VP.MIN_N))); memory: Dict[str, Dict[int, str]] = {}
    rec = dict(cfg.get("recurring") or {}); rec_share = float(rec.get("share", 0.0)); recent: deque = deque(maxlen=int(rec.get("recent_n", 50)))
    fsc = dict(cfg.get("fun_select") or {}); fun_sel = bool(fsc.get("enabled", False))     # azr19: buffer draws weighted by game fun
    if fun_sel and not fun:
        raise ValueError("train.fun_select needs vote.fun on (the score is the players' fun)")
    fscores = FS.FunScores(float(fsc.get("ema", FS.EMA))); fs_share = float(fsc.get("share_uniform", FS.SHARE_UNIFORM))
    fs_power = float(fsc.get("power", FS.POWER)); fs_refs = fun_sel and bool(fsc.get("refs", True)); fs_offers = fun_sel and bool(fsc.get("offers", True))

    def draw(pool):
        return FS.pick(pool, rng, fscores, fs_share, fs_power)
    if (predict or rec) and int(cfg["offerings"]) > int(cfg["worlds"]):
        raise ValueError(f"offerings {cfg['offerings']} > worlds {cfg['worlds']}: two offerings would share a world in one update")
    pf = open(out_dir / "predictions.jsonl", "a") if predict else None
    # predict.adapter Q (default): a separate predictor adapter (role Q, own optimiser group and clip), prompted with the seat's type latch.
    # The smoke 2026-10-19 trained predictions inside the A / W adapters: their grad norms went 0.3-1.3 -> 4-11, so the per-role clip shrank
    # the play/vote RL gradient 5-10x. 'own' keeps that version.
    pred_who = None if prc.get("adapter", "Q") == "own" else "cur:" + PRED_ROLE
    if predict and pred_who and not (nets.roles and PRED_ROLE in nets.roles):
        raise ValueError(f"predict.adapter {PRED_ROLE}: add {PRED_ROLE} to train.extra_roles (or set predict.adapter: own)")
    seed = int(spec["seed"]); rng = random.Random(seed); out_dir.mkdir(parents=True, exist_ok=True)
    lot_rng = random.Random(seed * 1_000_033 + 21)                      # azr21: the lottery's own stream (leaves rng's sequence alone)
    buf = AZ.Buffer(seeds, cfg.get("max_ref_chars")); fs = VR.FunState(); G = int(cfg["group"]); P = int(cfg["proposals"])
    Eo = int(cfg["offerings"]); anchor_every = int(cfg.get("anchor_every") or 0)
    worlds = [{"w": EC.World(seed=seed * 1000 + k), "voters": [VR.Voter(base=d_base, decay=d_decay) for _ in range(SB.N_PLAYERS)]
               + [VR.Voter(base=b_base, decay=d_decay) for _ in range(max(n_bv, SB.N_BYSTANDERS))], "resets": [],
               "types": list(table_types[k % len(table_types)]) if table_types else [PLAYER] * SB.N_PLAYERS} for k in range(int(cfg["worlds"]))]
    groups = nets.role_params(); lr = float(cfg["lr"])
    ref_who = "cur:" + REF_ROLE if (nets.roles and REF_ROLE in nets.roles) else "base"   # azr14: proposer KL to the frozen warm copy
    p_named = [(n_, q) for n_, q in nets.model.named_parameters() if nets.roles and f".{A3.ROLE_ADAPTERS['P']}." in n_] if rb else []
    rb_hist: List[float] = []; rb_best = -1.0; rb_state: Optional[Dict[str, torch.Tensor]] = None; n_rollback = 0
    opt = torch.optim.AdamW([{"params": ps, "lr": float(cfg.get("lr_proposer", lr)) if r == "P" else (float(cfg.get("lr_predictor", lr)) if r == PRED_ROLE else lr)}
                             for r, ps in groups.items()], weight_decay=0.0)
    tel = open(out_dir / "train_telemetry.jsonl", "a"); props = open(out_dir / "proposals.jsonl", "a"); trf = open(out_dir / "traces.jsonl", "a")
    att = open(out_dir / "proposer_attempts.jsonl", "a")
    base_cache: Dict[bytes, torch.Tensor] = {}; save_every = int(cfg.get("save_every") or 0)
    t0 = time.time(); done = 0; stopped = "steps"; bar = tqdm(total=n_steps, desc=f"vote {spec['name']}", unit="step")
    for step in range(n_steps):
        if max_minutes is not None and (time.time() - t0) / SECONDS_PER_MINUTE > float(max_minutes):
            stopped = "time"; break
        ts = time.time(); gen = torch.Generator().manual_seed(seed * 1_000_003 + step); tim: Counter = Counter(); cnt: Counter = Counter()
        fun_bl = {t: (fs.fun_baseline(t) if centre else 0.0) for t in (PLAYER, WINNER)}   # frozen for the update: every replay centred alike
        # ---------------------------------------------------------------- 1. propose
        if fs_refs:                                                     # azr19: references drawn by fun (half uniform)
            rpool = [it for it in buf.items if len(it["code"]) <= buf.max_ref_chars] or buf.items if buf.max_ref_chars else buf.items
            refs = [[draw(rpool)["code"] for _ in range(int(cfg["n_refs"]))] for _ in range(P)]
        else:
            refs = [[r["code"] for r in buf.sample(int(cfg["n_refs"]), rng, refs=True)] for _ in range(P)]
        ptexts = [PB.prompt_text(rf, p_variant) for rf in refs]
        prompts = [AZ.chat_ids(tok, t, PROPOSER_SYSTEM3) for t in ptexts]
        known = {AZ._norm(it["code"]) for it in buf.items}
        codes: List[Optional[str]] = [None] * P; texts = [""] * P; verdicts: List[Dict] = [{"ok": False, "reason": "no_code", "detail": ""} for _ in range(P)]
        samples: List[Dict] = []; hist: List[List[Dict]] = [[] for _ in range(P)]; rnd_of = [0] * P; todo_p = list(range(P))
        for rnd in range(p_repairs + 1):                                 # azr13: round 0 = the proposal, then repair rounds on failures
            if not todo_p:
                break
            ids = ([prompts[i] for i in todo_p] if rnd == 0 else
                   [PB.chat_ids_messages(tok, PB.repair_messages(PROPOSER_SYSTEM3, ptexts[i], hist[i])) for i in todo_p])
            with nets.use("cur:P") as m:
                gen_ = AZ.generate(m, tok, ids, int(cfg["max_new_tokens"]), p_temp, seed * 7919 + step + 100003 * rnd,
                                   int(cfg["gen_batch"]) if rnd == 0 else int(pc.get("repair_batch", cfg["gen_batch"])))
            txt_ = [tok.decode(c, skip_special_tokens=True) for c in gen_]; cod_ = [AZ.extract_code(t) for t in txt_]
            if p_fix:
                cod_ = [PB.fix_imports(c) if c else c for c in cod_]
            degen = [p_degen and PB.degenerate(t) for t in txt_]            # azr14: repetition loops score as empty output
            live = [k for k, c in enumerate(cod_) if c is not None and not degen[k] and AZ._norm(c) not in known]
            vm_ = dict(zip(live, pool.validate_many([cod_[k] for k in live], seed=step + 7 * rnd, readouts=True, affordance=False,
                                                    require_effect=require_effect) if live else []))
            nxt = []
            for k, i in enumerate(todo_p):
                v = vm_.get(k) or ({"ok": False, "reason": "degenerate", "detail": "repetition loop"} if degen[k] else
                                   {"ok": False, "reason": "copy", "detail": ""} if cod_[k] is not None else {"ok": False, "reason": "no_code", "detail": ""})
                codes[i], texts[i], verdicts[i], rnd_of[i] = cod_[k], txt_[k], v, rnd
                samples.append({"i": i, "round": rnd, "ids": ids[k], "comp": gen_[k], "verdict": v, "chars": len(cod_[k] or txt_[k])})
                hist[i].append({"text": txt_[k], "reason": v.get("reason"), "detail": str(v.get("detail", ""))[:PB.DETAIL_CHARS]})
                if not v["ok"]:
                    nxt.append(i)
            todo_p = nxt
        valid = [i for i in range(P) if verdicts[i]["ok"]]
        tim["propose"] = time.time() - ts; t_ = time.time()
        # ---------------------------------------------------------------- 2. offerings
        offers = []
        for j in range(Eo):
            wd = worlds[(step * Eo + j) % len(worlds)]
            if j < len(valid):
                i = valid[j]; g = {"code": codes[i], "rules": verdicts[i]["stats"]["rules"], "name": f"s{seed}_step{step}_p{i}", "source": "proposal", "pidx": i,
                                   "stats": {"agency": verdicts[i]["stats"].get("agency"), "agency_strict": verdicts[i]["stats"].get("agency_strict")}}
            elif anchors and anchor_every and step % anchor_every == 0 and j == Eo - 1:
                g = dict(anchors[(step // anchor_every) % len(anchors)], source="anchor")
            else:
                from_recent = bool(recent) and rng.random() < rec_share        # azr15: recently added proposer games come back
                if fs_offers:                                           # azr19: buffer / recurring offers drawn by fun (half uniform)
                    g = dict(draw(list(recent)) if from_recent else draw(buf.items))
                else:
                    g = dict(rng.choice(list(recent)) if from_recent else buf.sample(1, rng)[0])
                g.setdefault("name", g.get("source", "buffer"))
                g["source"] = g.get("source", "buffer"); g["recent"] = from_recent
            offers.append({"j": j, "wd": wd, "g": g})
        # per (offering, replay) episode state
        kn = ctl.knobs() if ctl else VC.fixed_knobs()                   # this update's weights (azr11); fixed runs: the azr10 rules
        veto_pen_k, veto_vpen_k = veto_pen * kn["taker_scale"], veto_vpen * kn["group_scale"]
        growth_k = None if growth is None else float(growth) * kn["growth_scale"]
        w_A_k = VP.scaled(w_A if w_A is not None else VR.FUN_WEIGHTS, kn.get("fun_scale", 1.0) if focus == PLAYER else 1.0)   # azr15: the mixer
        w_W_k = VP.scaled(w_W, kn.get("fun_scale", 1.0)) if focus == WINNER else w_W   # scales the focus type's fun weights (azr17: W if focus W)
        w_k = {PLAYER: w_A_k, WINNER: w_W_k}
        eps = [{"o": o, "r": r, "w": copy.deepcopy(o["wd"]["w"]), "voters": copy.deepcopy(o["wd"]["voters"]), "resets": list(o["wd"]["resets"]),
                "types": list(o["wd"]["types"])} for o in offers for r in range(G)]
        for e in eps:                                                    # azr11: the players' decline reward / cost this update
            for s_ in range(SB.N_PLAYERS):                               # (azr12: A seats only; winners keep the plain decline reward)
                if e["types"][s_] == focus:                               # azr17: the focus type (A unless focus W)
                    e["voters"][s_].base = d_base + kn["decline_bonus"]; e["voters"][s_].cost = kn["decline_cost"]
        D: List[Dict] = []                         # every LLM decision: prompt, n, choice, system, who, key (offering, phase, seat), replay
        R: Dict[Tuple, float] = {}                 # returns per (offering j, phase, seat, replay)

        def run_votes(phase: str):
            """Collect votes for every episode still running; fills ep[phase] = list of votes per voter (players 0-2, bystanders 3-4)."""
            items = []
            for e in eps:
                if phase == "v2" and not e.get("played1"):
                    e["v2"] = None; continue
                w = e["w"]; rules = e["o"]["g"]["rules"]; can_reset = _resets_left(e) > 0
                popts = [VR.PLAY, VR.DECLINE] + ([VR.RESET] if can_reset else [])
                for s in range(SB.N_PLAYERS):
                    items.append((e, s, False, popts))
                if bys_vote:
                    for b in range(n_bv):
                        items.append((e, SB.N_PLAYERS + b, True, [VR.PLAY, VR.DECLINE]))
            if not voting:
                for e in eps:
                    e[phase] = None if (phase == "v2" and not e.get("played1")) else [VR.PLAY] * (SB.N_PLAYERS + n_bv)
                return
            by_who = defaultdict(list)
            for it in items:
                by_who[f"cur:{BYSTANDER if it[2] else it[0]['types'][it[1]]}"].append(it)
            for e in eps:
                if not (phase == "v2" and not e.get("played1")):
                    e[phase] = [None] * (SB.N_PLAYERS + n_bv)
            for who, its in by_who.items():
                pr = [vote_prompt(e["o"]["g"]["rules"], e["w"], (s - SB.N_PLAYERS) // b_seats if byst else s, byst, phase == "v2",
                                  e.get("last") if phase == "v2" else None, [(BYSTANDER_OPTIONS if byst else PLAYER_OPTIONS)[o] for o in opts],
                                  veto=veto, note=VETO_NOTE if (veto and byst and phase == "v2" and (e.get("g1_harmed") or [False] * SB.N_BYSTANDERS)[s - SB.N_PLAYERS]) else None,
                                  types=e["types"] if table_types else None, extra=None if byst else player_extra(e, s, phase))
                      for e, s, byst, opts in its]
                sy = [BYSTANDER_SYSTEM if byst else latch(e["types"][s]) for e, s, byst, _ in its]
                for q_, (e, s, byst, opts) in enumerate(its):              # azr15: drop the extra text if it would overflow max_len
                    if not byst and player_extra(e, s, phase) and not _fits(pol, pr[q_], sy[q_]):
                        pr[q_] = vote_prompt(e["o"]["g"]["rules"], e["w"], s, False, phase == "v2", e.get("last") if phase == "v2" else None,
                                             [PLAYER_OPTIONS[o] for o in opts], veto=veto, types=e["types"] if table_types else None); cnt["extra_dropped"] += 1
                with nets.use(who):
                    probs = pol.probs(pr, [len(it[3]) for it in its], systems=sy)
                for (e, s, byst, opts), p_, prompt, system in zip(its, probs, pr, sy):
                    k = PO.sample(p_, gen); e[phase][s] = opts[k]
                    D.append({"prompt": prompt, "n": len(opts), "k": k, "sys": system, "who": who, "key": (e["o"]["j"], phase, s), "r": e["r"]})

        def play(phase: str):
            vk = "v1" if phase == "g1" else "v2"; inst, owners = [], []
            for e in eps:
                votes = e.get(vk)
                if votes is None:
                    e[f"played{phase[-1]}"] = False; continue
                if voting and lot is not None:                           # azr21: lottery vote
                    go, reset, e[f"p_play{phase[-1]}"] = VR.lottery(votes, SB.N_PLAYERS, w_lot, lot_rng.random())
                else:
                    go, reset = VR.majority(votes) if voting else (True, False)
                if phase == "g2" and veto and e.get("g1_harmed"):            # azr10: a harmed bystander's refusal blocks the rematch alone
                    vet = VR.harm_veto(votes, e["g1_harmed"], SB.N_PLAYERS)
                    if vet:
                        go = False; e["vetoed"] = vet
                e[f"played{phase[-1]}"] = go; e[f"reset{phase[-1]}"] = reset and go
                for s, v in enumerate(votes):                             # decline streaks (players and bystanders)
                    e[f"{phase}_dr{s}"] = e["voters"][s].vote(v)
                if not go:
                    continue
                w = e["w"]
                if reset:
                    w.wallets = [EC.START_WALLET] * SB.N_PLAYERS; e["resets"].append(w.games)
                endow = w.endowments(); e[f"{phase}_snap"] = (copy.deepcopy(w), list(endow), list(w.bystanders))
                seats = [LLM if votes[s] in (VR.PLAY, VR.RESET) else RESTRAINED for s in range(SB.N_PLAYERS)]
                inst.append({"code": e["o"]["g"]["code"], "rules": e["o"]["g"]["rules"], "endowments": list(endow), "bystanders": list(w.bystanders),
                             "seed": rng.randrange(2 ** 31), "seats": seats, "types": list(e["types"]), "who": ["cur"] * SB.N_PLAYERS,
                             "public": True, "world": world_text(w, veto) + (WINNER_NOTE if WINNER in e["types"] else "")})
                owners.append(e)
            if not inst:
                return
            results, decs, masses, _, traces = A3.play_batch3(server, pol, nets, inst, rng, gen, think=None)
            for (i, p, prompt, n, k, sy, who) in decs:
                e = owners[i]; D.append({"prompt": prompt, "n": n, "k": k, "sys": sy, "who": who, "key": (e["o"]["j"], phase, p), "r": e["r"]})
            for e, x, res, tr in zip(owners, inst, results, traces):
                e[f"{phase}_res"] = res; e[f"{phase}_trace"] = tr; e[f"{phase}_inst"] = x

        def score(phase: str):
            for e in eps:
                if not e.get(f"played{phase[-1]}") or "error" in e.get(f"{phase}_res", {"error": 1}):
                    if e.get(f"played{phase[-1]}"):
                        e[f"played{phase[-1]}"] = False; e[f"{phase}_err"] = True
                    continue
                snap, endow, bys_in = e[f"{phase}_snap"]; out = e[f"{phase}_res"]["out"]; x = e[f"{phase}_inst"]; tr = e[f"{phase}_trace"]
                ev = EC.game_outcome(snap, out, 0); gains = EC.rewards("finite", endow, ev); name = e["o"]["g"]["name"]
                terms = []
                for s in range(SB.N_PLAYERS):
                    if x["seats"][s] != LLM:
                        terms.append(None); continue
                    mv = [m for m in tr if m["player"] == s]
                    ag = (e["o"]["g"].get("stats") or {}).get(ag_key) if fun_v2 else None
                    terms.append(VR.own_terms(fs, name, e["types"][s], gains[s], [m["n_moves"] for m in mv], [m.get("describe", "") for m in mv],
                                              record_novelty=e["r"] == 0, agency=(0.0 if ag is None else ag) if fun_v2 else None))
                rew, full = VR.seat_rewards(gains, terms, weights=w_A_k, fun=fun, baseline=fun_bl[PLAYER])
                if WINNER in e["types"]:                                 # azr12: winners score their own fun only, against their own baseline
                    rw_w, fl_w = VR.seat_rewards(gains, terms, weights=w_W_k, fun=fun, baseline=fun_bl[WINNER])
                    rew = [rw_w[s] if e["types"][s] == WINNER else rew[s] for s in range(SB.N_PLAYERS)]
                    full = [fl_w[s] if e["types"][s] == WINNER else full[s] for s in range(SB.N_PLAYERS)]
                e[f"{phase}_gains"] = gains; e[f"{phase}_rew"] = rew; e[f"{phase}_terms"] = full; e[f"{phase}_ev"] = ev
                scale = sum(endow) / len(endow)
                e[f"{phase}_bys"] = [(VR.bystander_share_reward(b0, b1) if growth is None else VR.plant_share_reward(b0, b1, growth_k)) if b_share
                                     else (VR.bystander_reward(b0, b1, scale) if growth is None else VR.plant_reward(b0, b1, scale, growth_k))
                                     for b0, b1 in zip(bys_in, out["bystanders"])]
                e["last"] = _summary(out, endow, bys_in)
                if phase == "g1":
                    e["g1_harmed"] = VR.harmed(bys_in, out["bystanders"]); e["g1_takers"] = VR.takers(tr, SB.N_BYSTANDERS)
                EC.settle(e["w"], endow, out, stomp_mode="persist")
                if e["r"] == 0 and phase == "g1" and use_mem:                  # azr15: what each seat last saw of this game
                    remember(name, endow, bys_in, out, tr, [x["seats"][s] == LLM for s in range(SB.N_PLAYERS)], trial=False)
                if e["r"] == 0:
                    won = VR.winners(gains, [t is not None for t in terms])
                    for s in range(SB.N_PLAYERS):
                        if terms[s] is not None:
                            t_ = e["types"][s]; fs.update(name, t_, gains[s])
                            if fun_v2:
                                fs.update_win(name, t_, bool(won[s]))
                            if centre:
                                fs.update_fun(t_, VR.fun_total(full[s], w_k[t_]))

        def remember(name, endow, bys_in, out, tr, llm, trial):
            gn = [r1 - r0 for r0, r1 in zip(endow, out["resources"])]; won = VR.winners(gn, llm)
            for s in range(SB.N_PLAYERS):
                if llm[s]:
                    memory.setdefault(name, {})[s] = VP.memory_note(s, [str(m.get("move")) for m in tr if m["player"] == s], endow[s], out["resources"][s],
                                                                    bys_in, out["bystanders"], won[s], trial,
                                                                    *((list(endow), list(out["resources"])) if mem_others else (None, None)))

        def run_trials(tro) -> int:
            """azr15: each new game played once in a scratch copy of its world by the current players; experience only (not trained)."""
            inst = []
            for o in tro:
                w = copy.deepcopy(o["wd"]["w"]); endow = w.endowments()
                inst.append({"code": o["g"]["code"], "rules": o["g"]["rules"], "endowments": list(endow), "bystanders": list(w.bystanders),
                             "seed": rng.randrange(2 ** 31), "seats": [LLM] * SB.N_PLAYERS, "types": list(o["wd"]["types"]), "who": ["cur"] * SB.N_PLAYERS,
                             "public": True, "world": world_text(w, veto) + (WINNER_NOTE if WINNER in o["wd"]["types"] else "")})
            if not inst:
                return 0
            results, _, _, _, traces = A3.play_batch3(server, pol, nets, inst, rng, gen, think=None)
            n_ = 0
            for o, x, res, tr in zip(tro, inst, results, traces):
                if "error" in res:
                    continue
                remember(o["g"]["name"], x["endowments"], x["bystanders"], res["out"], tr, [True] * SB.N_PLAYERS, trial=True); n_ += 1
            return n_

        def score_prompts(q) -> List[List[float]]:
            """q: (offer, seat, role, part, edges, prompt); band probabilities per item, scored by the seat type's adapter."""
            out_: List[Optional[List[float]]] = [None] * len(q)
            for who in sorted({pred_who or f"cur:{x[2]}" for x in q}):
                idx = [k for k, x in enumerate(q) if (pred_who or f"cur:{x[2]}") == who]
                with nets.use(who):
                    pp = pol.probs([q[k][5] for k in idx], [VP.N_BINS] * len(idx), systems=[latch(q[k][2]) for k in idx])
                for k, p_ in zip(idx, pp):
                    out_[k] = p_
            return out_

        def make_predictions() -> None:
            """azr15: part bands, then the total band from the mix + the seat's own part bands (most likely band each)."""
            items = [(o, s, o["wd"]["types"][s]) for o in offers for s in range(SB.N_PLAYERS) if binner.ready(o["wd"]["types"][s], VP.TOTAL)]
            heads = {(o["j"], s): game_head(o["g"]["rules"], o["wd"]["w"], s, o["wd"]["types"] if table_types else None, veto,
                                            memory.get(o["g"]["name"], {}).get(s) if use_mem else None) for o, s, _ in items}
            q = [(o, s, t_, part, ed, VP.part_prompt(heads[(o["j"], s)], part, t_, ed))
                 for o, s, t_ in items for part in VP.COMPONENTS for ed in [binner.edges(t_, part)] if ed is not None]
            q = [x for x in q if _fits(pol, x[5], latch(x[2]))]
            parts_: Dict[Tuple[int, int], Dict[str, Tuple[int, List[float]]]] = defaultdict(dict)
            for (o, s, t_, part, ed, pr), p_ in zip(q, score_prompts(q)):
                b = max(range(len(p_)), key=lambda i_: p_[i_]); parts_[(o["j"], s)][part] = (b, ed)
                PD.append({"prompt": pr, "n": VP.N_BINS, "sys": latch(t_), "who": pred_who or f"cur:{t_}", "key": (o["j"], s, part), "edges": ed, "band": b})
            q2 = []
            for o, s, t_ in items:
                ed = binner.edges(t_, VP.TOTAL); wts = w_k[t_]
                q2.append((o, s, t_, VP.TOTAL, ed, VP.total_prompt(heads[(o["j"], s)], t_, wts, fun_bl[t_], parts_[(o["j"], s)], ed)))
            q2 = [x for x in q2 if _fits(pol, x[5], latch(x[2]))]
            for (o, s, t_, part, ed, pr), p_ in zip(q2, score_prompts(q2)):
                b = max(range(len(p_)), key=lambda i_: p_[i_]); wts = w_k[t_]; key = (o["j"], s)
                vals = {p: binner.centres(t_, p, e_)[bb] for p, (bb, e_) in parts_[key].items()}
                preds[key] = {"role": t_, "total_band": b, "total_probs": p_, "total_edges": ed, "exp_total": VP.expected(p_, binner.centres(t_, VP.TOTAL, ed)),
                              "composed": VP.composed(vals, wts, fun_bl[t_]), "parts": {p: bb for p, (bb, _) in parts_[key].items()}}
                PD.append({"prompt": pr, "n": VP.N_BINS, "sys": latch(t_), "who": pred_who or f"cur:{t_}", "key": (o["j"], s, VP.TOTAL), "edges": ed, "band": b})

        def player_extra(e, s, phase) -> Optional[str]:
            if phase != "v1" or not predict:
                return None
            bits = [memory.get(e["o"]["g"]["name"], {}).get(s)] if use_mem else []
            pd_ = preds.get((e["o"]["j"], s))
            if pd_:
                v_ = e["voters"][s]
                bits.append(VP.vote_note(pd_["total_band"], pd_["total_edges"], VR.decline_reward(v_.streak, v_.base, v_.decay) - v_.cost))
            return " ".join(b for b in bits if b) or None

        PD: List[Dict] = []; preds: Dict[Tuple[int, int], Dict] = {}; n_trial = 0     # azr15
        if trial:
            tro = [o for o in offers if o["g"]["source"] == "proposal"]
            if not tro and prc.get("trial_seed_when_empty") and offers:    # smoke only: exercise the trial on a buffer game
                tro = [offers[0]]
            n_trial = run_trials(tro); tim["trial"] = time.time() - t_; t_ = time.time()
        if predict:
            make_predictions(); tim["predict"] = time.time() - t_; t_ = time.time()
        run_votes("v1"); tim["vote"] += time.time() - t_; t_ = time.time()
        play("g1"); score("g1"); tim["play"] += time.time() - t_; t_ = time.time()
        run_votes("v2"); tim["vote"] += time.time() - t_; t_ = time.time()
        play("g2"); score("g2"); tim["play"] += time.time() - t_; t_ = time.time()
        # ---------------------------------------------------------------- 3. returns (Pyrrhic rule on game 1)
        n_pyr = 0; realised: Dict[Tuple[int, int], List[Dict[str, float]]] = defaultdict(list); role_of: Dict[Tuple[int, int], str] = {}
        fun_rows: List[Tuple[float, float]] = []
        for e in eps:
            j, r = e["o"]["j"], e["r"]
            social_before = list(e["g1_rew"]) if e.get("played1") else None   # azr12: winners are exempt from the social rules
            if e.get("played1") and pyr and e.get("v2"):
                v2p = [e["v2"][s] if e["g1_inst"]["seats"][s] == LLM else None for s in range(SB.N_PLAYERS)]
                e["g1_rew"], e["pyrrhic"] = VR.pyrrhic(e["g1_rew"], e["g1_gains"], v2p, strength=kn["group_scale"]); n_pyr += int(e["pyrrhic"])
            if e.get("played1") and e.get("vetoed"):                          # azr10: the vetoing bystanders' takers pay
                e["g1_rew"] = VR.veto_penalties(e["g1_rew"], e["g1_takers"], e["vetoed"], veto_pen_k)
            if social_before is not None:
                e["g1_rew"] = VR.exempt(e["g1_rew"], social_before, e["types"], WINNER)
                for s in range(SB.N_PLAYERS):                              # azr15: the realised parts of each played LLM seat
                    if e["g1_inst"]["seats"][s] != LLM or e["g1_rew"][s] is None or social_before[s] is None:
                        continue
                    pv = VP.parts_of(e["g1_gains"][s], e["g1_rew"][s] - social_before[s], e["g1_terms"][s]); pv[VP.TOTAL] = e["g1_rew"][s]
                    realised[(j, s)].append(pv); role_of[(j, s)] = e["types"][s]
                    if e["types"][s] == focus and fun:
                        fun_rows.append((VR.fun_total(e["g1_terms"][s], w_k[focus]) - fun_bl[focus], pv["gain"] + pv["social"]))
                        if fun_sel:                                     # azr19: the game's fun score, base weights (no fun scale)
                            fscores.update(e["o"]["g"]["name"], VR.fun_total(e["g1_terms"][s], w_base[focus]))
            for ph, gph in (("v1", "g1"), ("v2", "g2")):
                votes = e.get(ph)
                if votes is None:
                    continue
                for s, v in enumerate(votes):
                    if s < SB.N_PLAYERS:
                        val = e[f"{gph}_dr{s}"] if v == VR.DECLINE else ((e[f"{gph}_rew"][s] or 0.0) if e.get(f"played{gph[-1]}") else 0.0)
                        if ph == "v2" and veto and e.get("vetoed") and e["types"][s] != WINNER:   # azr10: voting for a vetoed rematch is paid for
                            val -= VR.rematch_voter_penalties(votes, e["vetoed"], SB.N_PLAYERS, veto_vpen_k)[s]
                    else:
                        b = (s - SB.N_PLAYERS) // b_seats                     # azr20: every seat of a bystander earns its reward
                        val = e[f"{gph}_dr{s}"] if v == VR.DECLINE else (e[f"{gph}_bys"][b] if e.get(f"played{gph[-1]}") else 0.0)
                    R[(j, ph, s, r)] = val
            for gph in ("g1", "g2"):
                if e.get(f"played{gph[-1]}"):
                    for s in range(SB.N_PLAYERS):
                        if e[f"{gph}_rew"][s] is not None:
                            R[(j, gph, s, r)] = e[f"{gph}_rew"][s]
        # advantages: return minus the mean over the replays of the same (offering, phase, seat) that have one
        means = {}
        for (j, ph, s, r), v in R.items():
            means.setdefault((j, ph, s), []).append(v)
        means = {k: sum(v) / len(v) for k, v in means.items() if len(v) >= 2}
        for d in D:
            j, ph, s = d["key"]; v = R.get((j, ph, s, d["r"])); mu = means.get((j, ph, s))
            d["adv"] = None if v is None or mu is None else v - mu
        for d in PD:                                                     # azr15: targets = the bands the real game 1s landed in
            j, s, part = d["key"]; d["targets"] = [VP.band(v[part], d["edges"]) for v in realised.get((j, s), [])]
        if predict:
            for key, vals in realised.items():
                for v in vals:
                    for part, x in v.items():
                        binner.add(role_of[key], part, x)
        # ---------------------------------------------------------------- 4. proposer reward
        bonus_of: Dict[int, float] = {}
        if voting and vote_all and valid:                                # azr14: every valid game gets the players' verdict (no play)
            todo_v = []
            for i in valid:                                              # azr15: the type-A seats' expected predicted total, when predicted
                o = next((o for o in offers if o["g"].get("pidx") == i), None)
                ex_ = [preds[(o["j"], s)]["exp_total"] for s in range(SB.N_PLAYERS)
                       if o is not None and (o["j"], s) in preds and preds[(o["j"], s)]["role"] == focus]
                if ex_:
                    bonus_of[i] = sum(ex_) / len(ex_); verdicts[i]["vote_bonus"] = bonus_of[i]; verdicts[i]["bonus_kind"] = "predicted"
                else:
                    todo_v.append(i)
            pv = vote_probe(nets, pol, [verdicts[i]["stats"]["rules"] for i in todo_v], EC.World(seed=seed * 1000 + 999 + step), veto,
                            [focus] * SB.N_PLAYERS if table_types else None, focus) if todo_v else []
            for i, b in zip(todo_v, pv):
                bonus_of[i] = b; verdicts[i]["vote_bonus"] = b; verdicts[i]["bonus_kind"] = "p_play"
        if voting and vote_all and not valid and pc.get("probe_seed_when_empty"):   # smoke only: exercise the probe on a seed game
            verdicts_probe = vote_probe(nets, pol, [buf.items[0]["rules"]], EC.World(seed=seed * 1000 + 999 + step), veto,
                                        [focus] * SB.N_PLAYERS if table_types else None, focus)
            tim_probe_seed = verdicts_probe[0]
        else:
            tim_probe_seed = None
        for i in ([] if vote_all else valid):
            if voting:
                o = next((o for o in offers if o["g"].get("pidx") == i), None)
                if o is not None:
                    ev_ = [e for e in eps if e["o"] is o]
                    bonus_of[i] = VR.proposer_vote_bonus([v for e in ev_ for v in (e.get("v1") or [])], [v for e in ev_ for v in (e.get("v2") or [])])
                    verdicts[i]["vote_bonus"] = bonus_of[i]
        for smp in samples:                                             # every attempt is a sample, scored by its own verdict
            rw = PB.proposer_reward(smp["verdict"], smp["chars"], pc)
            if smp["verdict"]["ok"] and smp["i"] in bonus_of:
                rw += VOTE_BONUS * bonus_of[smp["i"]]
            smp["rew"] = rw
        p_rew = [next(s_["rew"] for s_ in reversed(samples) if s_["i"] == i) if any(s_["i"] == i for s_ in samples) else 0.0 for i in range(P)]
        # ---------------------------------------------------------------- 5. update
        opt.zero_grad(set_to_none=True); nets.model.train(); kl_p = 0.0; n_p = 0
        base_r = {r_: sum(s_["rew"] for s_ in samples if s_["round"] == r_) / max(1, sum(s_["round"] == r_ for s_ in samples))
                  for r_ in {s_["round"] for s_ in samples}}                # baseline: mean over the attempts of the same round
        for smp in samples:
            adv = smp["rew"] - base_r[smp["round"]]
            if abs(adv) < 1e-9 or not smp["comp"]:
                continue
            with torch.no_grad(), nets.use(ref_who) as m:
                lr_ = AZ.batch_completion_logprob(m, [smp["ids"]], [smp["comp"]], tok.pad_token_id)[0]
            with nets.use("cur:P") as m:
                lp = AZ.batch_completion_logprob(m, [smp["ids"]], [smp["comp"]], tok.pad_token_id)[0]
                d_ = lr_ - lp; kl = (torch.exp(d_) - d_ - 1).mean()
                ((-adv * lp.mean() + float(cfg["beta_proposer"]) * kl) / max(1, len(samples))).backward()
            kl_p += float(kl); n_p += 1
        sd = sorted([d for d in D if d["adv"] is not None and abs(d["adv"]) > 0], key=lambda d: d["who"]); n_all = max(1, len(sd))
        kl_role: Dict[str, List[float]] = defaultdict(list)
        mbs = [part[s0:s0 + pol.micro_batch] for part in [[d for d in sd if d["who"] == w] for w in sorted({d["who"] for d in sd})]
               for s0 in range(0, len(part), pol.micro_batch)]
        for mb in mbs:
            pr_, n_, sy_ = [d["prompt"] for d in mb], [d["n"] for d in mb], [d["sys"] for d in mb]
            t_b = time.time(); refl = A3.base_logp(pol, nets, base_cache, pr_, n_, sy_, cnt); tim["base_ref"] += time.time() - t_b
            with nets.use(mb[0]["who"]):
                cur = pol.forward(pr_, n_, systems=sy_)
                ch = torch.tensor([d["k"] for d in mb], device=cur["logp"].device)
                a = torch.tensor([d["adv"] for d in mb], device=cur["logp"].device, dtype=cur["logp"].dtype)
                ke = PO.kl_and_entropy(cur["logp"], refl.to(cur["logp"].device, cur["logp"].dtype))
                ((-(a * cur["logp"].gather(1, ch[:, None]).squeeze(1)) + float(cfg["beta_solver"]) * ke["kl"]).sum() / n_all).backward()
            kl_role[mb[0]["who"]] += ke["kl"].detach().float().cpu().tolist()
        pl_loss: List[float] = []; pd_t = [d for d in PD if d.get("targets")]; n_pd = max(1, len(pd_t))   # azr15: supervised predictions
        for part_ in [[d for d in pd_t if d["who"] == w] for w in sorted({d["who"] for d in pd_t})]:
            for s0 in range(0, len(part_), pol.micro_batch):
                mb = part_[s0:s0 + pol.micro_batch]
                with nets.use(mb[0]["who"]):
                    cur = pol.forward([d["prompt"] for d in mb], [d["n"] for d in mb], systems=[d["sys"] for d in mb])
                    ce = torch.stack([-cur["logp"][i_, torch.tensor(d["targets"], device=cur["logp"].device)].mean() for i_, d in enumerate(mb)])
                    (pred_w * ce.sum() / n_pd).backward()
                pl_loss += ce.detach().float().cpu().tolist()
        gn_role = {}
        for r_, ps in groups.items():
            gs = [q.grad.detach().float().norm() for q in ps if q.grad is not None]
            gn_role[r_] = float(torch.norm(torch.stack(gs))) if gs else 0.0
            if gs:
                torch.nn.utils.clip_grad_norm_([q for q in ps if q.grad is not None], float(cfg["grad_clip"]))
        opt.step(); nets.model.eval(); done = step + 1; tim["update"] = time.time() - t_
        rb_event = None
        if rb and P:                                                     # azr14: keep the best proposer; roll back a collapse
            rb_hist.append(len(valid) / P); wdw = int(rb["window"])
            if len(rb_hist) >= wdw:
                cur_v = sum(rb_hist[-wdw:]) / wdw
                if cur_v > rb_best:
                    if cur_v >= rb_best + float(rb.get("save_margin", 0.02)):
                        nets.model.save_pretrained(str(out_dir / "best_proposer"), selected_adapters=[A3.ROLE_ADAPTERS["P"]])
                    rb_best = cur_v; rb_state = {n_: q.detach().to("cpu", copy=True) for n_, q in p_named}
                elif rb_state is not None and ((rb_best >= float(rb["min_best"]) and cur_v < float(rb["frac"]) * rb_best)
                                               or step == rb.get("force_step")):   # force_step: smoke only, exercises the restore
                    with torch.no_grad():
                        for n_, q in p_named:
                            q.copy_(rb_state[n_].to(q.device, q.dtype)); opt.state.pop(q, None)
                    rb_hist.clear(); n_rollback += 1; rb_event = cur_v
        if save_every and done % save_every == 0:
            nets.model.save_pretrained(str(out_dir / "checkpoints" / f"step_{done}"), selected_adapters=list(nets.roles.values()))
        for i in range(P):
            if verdicts[i]["ok"]:
                it_ = {"code": codes[i], "rules": verdicts[i]["stats"]["rules"], "name": f"s{seed}_step{step}_p{i}", "source": f"s{seed}_step{step}",
                       "stats": {k: v for k, v in verdicts[i]["stats"].items() if k != "rules"}}
                buf.add(it_); recent.append(it_)
            props.write(json.dumps({"step": step, "i": i, "reason": None if verdicts[i]["ok"] else verdicts[i]["reason"], "reward": p_rew[i],
                                    "vote_bonus": verdicts[i].get("vote_bonus"), "chars": len(codes[i] or ""), "code": codes[i],
                                    "detail": verdicts[i].get("detail", "")[:300], "round": rnd_of[i],
                                    "agency": (verdicts[i].get("stats") or {}).get("agency") if verdicts[i]["ok"] else None,
                                    "agency_strict": (verdicts[i].get("stats") or {}).get("agency_strict") if verdicts[i]["ok"] else None,
                                    "bonus_kind": verdicts[i].get("bonus_kind")}) + "\n")
        props.flush()
        if p_repairs:                                                    # azr13: every attempt (proposal + repairs)
            for smp in samples:
                att.write(json.dumps({"step": step, "i": smp["i"], "round": smp["round"], "ok": bool(smp["verdict"]["ok"]),
                                      "reason": smp["verdict"].get("reason"), "reward": smp["rew"], "chars": smp["chars"]}) + "\n")
            att.flush()
        # ---------------------------------------------------------------- 6. commit worlds (replay 0) + logs
        for o in offers:
            e0 = next(e for e in eps if e["o"] is o and e["r"] == 0)
            o["wd"]["w"], o["wd"]["voters"], o["wd"]["resets"] = e0["w"], e0["voters"], e0["resets"]
            if e0["w"].games and any(x < EC.MIN_ENTRY for x in e0["w"].wallets):
                pass                                                     # settle() already restarted a ruined world
        for e in eps:
            if e["r"] != 0:
                continue
            rec = {"step": step, "j": e["o"]["j"], "game": e["o"]["g"]["name"], "source": e["o"]["g"]["source"], "v1": e.get("v1"), "v2": e.get("v2"),
                   "played1": e.get("played1"), "played2": e.get("played2"), "reset1": e.get("reset1"), "reset2": e.get("reset2"), "pyrrhic": e.get("pyrrhic"),
                   "vetoed": e.get("vetoed"), "g1_harmed": e.get("g1_harmed"), "g1_takers": e.get("g1_takers"),
                   "p_play1": e.get("p_play1"), "p_play2": e.get("p_play2"),      # azr21: the lottery's play probability (None = majority)
                   "rules": e["o"]["g"]["rules"][:TRACE_CHARS]}
            for gph in ("g1", "g2"):
                if e.get(f"played{gph[-1]}"):
                    rec[gph] = {"gains": e[f"{gph}_gains"], "rewards": e[f"{gph}_rew"], "terms": e[f"{gph}_terms"], "bystander_loss": e[f"{gph}_ev"]["bystander_loss"],
                                "top_share": e[f"{gph}_ev"]["top_share"], "stomp": e[f"{gph}_ev"]["stomp_state"] if "stomp_state" in e[f"{gph}_ev"] else e[f"{gph}_ev"].get("stomp"),
                                "seats": e[f"{gph}_inst"]["seats"], "moves": [{k: m.get(k) for k in ("player", "seat", "move", "n_moves", "dbys")} for m in e[f"{gph}_trace"]]}
            trf.write(json.dumps(rec) + "\n")
        trf.flush()
        row = telemetry_row(step, spec, eps, verdicts, p_rew, D, ts, tim, cnt, kl_p, n_p, kl_role, gn_role, n_pyr, base_cache, pol, buf, worlds)
        row["fun_baseline"] = fun_bl[focus]                            # 2026-10-07 grid telemetry (azr17: the focus type's)
        row["focus"] = focus
        r0 = [s_ for s_ in samples if s_["round"] == 0]                  # azr13 proposer telemetry
        row.update({"valid_share_r0": (sum(s_["verdict"]["ok"] for s_ in r0) / len(r0)) if r0 else None,
                    "no_code_share": (sum(v.get("reason") == "no_code" for v in verdicts) / P) if P else None,
                    "repair_fixed": sum(1 for i in valid if rnd_of[i] > 0), "proposer_attempts": len(samples),
                    "proposer_chars": (sum(s_["chars"] for s_ in r0) / len(r0)) if r0 else None,
                    "degenerate_share": (sum(v.get("reason") == "degenerate" for v in verdicts) / P) if P else None,
                    "vote_bonus_mean": AZ._mean([v.get("vote_bonus") for v in verdicts if v.get("ok")]),
                    "proposer_agency": AZ._mean([(v.get("stats") or {}).get("agency") for v in verdicts if v.get("ok")]),
                    "proposer_agency_strict": AZ._mean([(v.get("stats") or {}).get("agency_strict") for v in verdicts if v.get("ok")]),
                    "rollbacks": n_rollback, "rollback_event": rb_event, "best_valid": rb_best if rb else None, "vote_probe_seed": tim_probe_seed})
        row.update({f"knob_{k}": v for k, v in kn.items()})            # 2026-10-12 azr11: the weights this update used
        p1 = [e for e in eps if e.get("played1")]                        # 2026-10-10 azr10 telemetry
        row["harm_g1_share"] = AZ._mean([float(any(e.get("g1_harmed") or [])) for e in p1]) if p1 else None
        if table_types:                                                  # azr12: the controllers read A seats only
            row["harm_any_g1_share"] = row["harm_g1_share"]
            row["harm_g1_share"] = AZ._mean([float(VR.a_harm(e.get("g1_takers") or [], e["types"], focus)) for e in p1]) if p1 else None
            row.update(winner_telemetry(eps))
        row["veto_share"] = AZ._mean([float(bool(e.get("vetoed"))) for e in p1]) if (p1 and veto) else None
        row["valid_effect"] = AZ._mean([v["stats"].get("effect") for v in verdicts if v.get("ok")])
        row["decline_share_v1"] = AZ._mean([float(v == VR.DECLINE) for e in eps for s_, v in enumerate((e.get("v1") or [])[:SB.N_PLAYERS])
                                            if e["types"][s_] == focus]) if voting else None
        row["bystander_reward"] = AZ._mean([x for e in eps for g in ("g1", "g2") if e.get(f"played{g[-1]}") for x in (e.get(f"{g}_bys") or [])]) if bys_vote else None
        row["fun_share"] = VP.fun_share(fun_rows)                        # azr15: the mixer's statistic (A seats, game 1)
        row["recurring_offers"] = sum(bool(o["g"].get("recent")) for o in offers); row["offers"] = len(offers)
        if fun_sel:                                                      # azr19: fun scores of the offered games vs every scored game
            row["fun_select_scored"] = len(fscores.score)
            row["fun_select_offer_score"] = AZ._mean([x for o in offers for x in [fscores.get(o["g"].get("name", ""))] if x is not None])
            row["fun_select_mean_score"] = AZ._mean(list(fscores.score.values()))
        if predict:
            pairs = [(preds[k], v) for k, vs in realised.items() if k in preds for v in vs]
            pt = [d for d in PD if d.get("targets")]
            row.update({"pred_ready": bool(preds), "pred_n": len(PD), "pred_trained": len(pt), "pred_loss": AZ._mean(pl_loss), "trials": n_trial,
                        "pred_acc_total": AZ._mean([float(p["total_band"] == VP.band(v[VP.TOTAL], p["total_edges"])) for p, v in pairs]),
                        "pred_acc_parts": AZ._mean([float(d["band"] == t) for d in pt if d["key"][2] != VP.TOTAL for t in d["targets"]]),
                        "pred_spearman_total": VP.spearman([p["exp_total"] for p, _ in pairs], [v[VP.TOTAL] for _, v in pairs]),
                        "pred_spearman_composed": VP.spearman([p["composed"] for p, _ in pairs], [v[VP.TOTAL] for _, v in pairs]),
                        "pred_exp_total_A": AZ._mean([p["exp_total"] for p in preds.values() if p["role"] == PLAYER]),
                        "pred_exp_total_focus": AZ._mean([p["exp_total"] for p in preds.values() if p["role"] == focus]),
                        "memory_games": len(memory), "extra_dropped": cnt["extra_dropped"]})
            for (j, s), p in preds.items():
                o = next(o for o in offers if o["j"] == j)
                pf.write(json.dumps({"step": step, "j": j, "seat": s, "role": p["role"], "game": o["g"]["name"], "source": o["g"]["source"],
                                     "recent": bool(o["g"].get("recent")), "agency": (o["g"].get("stats") or {}).get("agency"),
                                     "agency_strict": (o["g"].get("stats") or {}).get("agency_strict"),
                                     "memory": bool(memory.get(o["g"]["name"], {}).get(s)), "total_band": p["total_band"], "total_probs": p["total_probs"],
                                     "total_edges": p["total_edges"], "exp_total": p["exp_total"], "composed": p["composed"], "parts": p["parts"],
                                     "realised": realised.get((j, s), [])}) + "\n")
            pf.flush()
        if ctl is not None:                                              # azr11: feed this update's statistics to the controllers
            # (AFTER decline_share_v1 is computed: the first build fed it None and the exit/stay controllers never moved)
            lam = ctl.update({"decline": row.get("decline_share_v1"), "harm": row.get("harm_g1_share"), "allow": row.get("bystander_allow"),
                              VC.FUN_STAT: row.get("fun_share")})
            row.update({f"ctl_{n}": v for n, v in lam.items()}); row.update({f"ctl_smoothed_{k}": v for k, v in ctl.smoothed.items()})
        tel.write(json.dumps(row) + "\n"); tel.flush()
        bar.update(1); bar.set_postfix(valid=f"{row['valid_share']:.2f}", play=f"{row.get('play_share_v1') or 0:.2f}", bl=f"{row.get('bystander_loss') or 0:.2f}")
    bar.close(); tel.close(); props.close(); trf.close(); att.close()
    if pf is not None:
        pf.close()
    with open(out_dir / "buffer.jsonl", "w") as f:
        for it in buf.items:
            f.write(json.dumps({k: it[k] for k in ("code", "rules", "source", "name") if k in it}) + "\n")
    return {"steps_done": done, "stopped": stopped, "minutes": (time.time() - t0) / SECONDS_PER_MINUTE, "buffer": len(buf.items), "past_step": None}


@torch.no_grad()
def vote_probe(nets: A3.Nets, pol: PO.LetterPolicy, rules: Sequence[str], w: EC.World, veto: bool, types: Optional[Sequence[str]],
               ptype: str = PLAYER) -> List[float]:
    """azr14: what the trained A players think of each game without playing it: mean over the three seats of P(play) on a first offer in a
    fresh world (two options, no sampling, not trained on). The proposer's vote bonus for EVERY valid game: in azr13 only the first
    ``offerings`` valid games were offered and voted, the rest scored 0, so the bonus followed slot order, not the game. ``ptype`` (azr17):
    the player type asked (its adapter and latch); W in the finite-only run."""
    prompts = [vote_prompt(r, w, s, False, False, None, [PLAYER_OPTIONS[VR.PLAY], PLAYER_OPTIONS[VR.DECLINE]], veto=veto, types=types)
               for r in rules for s in range(SB.N_PLAYERS)]
    with nets.use("cur:" + ptype):
        probs = pol.probs(prompts, [2] * len(prompts), systems=[latch(ptype)] * len(prompts))
    return [sum(float(p[0]) for p in probs[k * SB.N_PLAYERS:(k + 1) * SB.N_PLAYERS]) / SB.N_PLAYERS for k in range(len(rules))]


def _share(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def winner_telemetry(eps: Sequence[Dict]) -> Dict:
    """azr12: what the A players do about the winners, and what the winners do. Votes per type and table kind (W present or all-A),
    reward / gain / bystander-taking per type, and per A seat in sanction games the share of its moves that sanction a W vs an A seat."""
    out: Dict = {}
    for ph in ("v1", "v2"):
        for tbl, cond in (("wtable", lambda e: WINNER in e["types"]), ("atable", lambda e: WINNER not in e["types"])):
            out[f"decline_A_{ph}_{tbl}"] = _share([float(v == VR.DECLINE) for e in eps if cond(e) for s, v in enumerate((e.get(ph) or [])[:SB.N_PLAYERS])
                                                    if e["types"][s] == PLAYER])
        out[f"decline_W_{ph}"] = _share([float(v == VR.DECLINE) for e in eps for s, v in enumerate((e.get(ph) or [])[:SB.N_PLAYERS]) if e["types"][s] == WINNER])
        out[f"bys_allow_{ph}_wtable"] = _share([float(v == VR.PLAY) for e in eps if WINNER in e["types"] for v in (e.get(ph) or [])[SB.N_PLAYERS:]])
        out[f"bys_allow_{ph}_atable"] = _share([float(v == VR.PLAY) for e in eps if WINNER not in e["types"] for v in (e.get(ph) or [])[SB.N_PLAYERS:]])
    acc: Dict[str, List[float]] = defaultdict(list)
    for e in eps:
        for g in ("g1", "g2"):
            if not e.get(f"played{g[-1]}"):
                continue
            x = e[f"{g}_inst"]; tr = e[f"{g}_trace"]; verbs = SN.player_verbs(e["o"]["g"]["rules"])
            for s in range(SB.N_PLAYERS):
                if x["seats"][s] != LLM:
                    continue
                t = e["types"][s]; mv = [m for m in tr if m["player"] == s]
                if e[f"{g}_rew"][s] is not None:
                    acc[f"reward_{t}"].append(e[f"{g}_rew"][s])
                acc[f"gain_{t}"].append(e[f"{g}_gains"][s])
                if mv:
                    acc[f"take_{t}"].append(sum(any(d < 0 for d in (m.get("dbys") or [])) for m in mv) / len(mv))
                if verbs and t == PLAYER and WINNER in e["types"]:
                    st = SN.seat_stats(tr, verbs, e["types"], s, SN.rule_of(e["o"]["g"]["rules"]))
                    if st["sanction_share"] is not None:
                        n_w, n_a = e["types"].count(WINNER), e["types"].count(PLAYER) - 1      # per available target of that type
                        acc["A_sanction_at_W"].append(st["sanction_at_W"] / n_w)
                        if n_a:
                            acc["A_sanction_at_A"].append(st["sanction_at_A"] / n_a)
                        acc["A_sanction_just_wtable"].append(st["sanction_just"])
    out.update({k: _share(v) for k, v in acc.items()})
    return out


def telemetry_row(step, spec, eps, verdicts, p_rew, D, ts, tim, cnt, kl_p, n_p, kl_role, gn_role, n_pyr, base_cache, pol, buf, worlds) -> Dict:
    m = AZ._mean; P = len(verdicts)
    pl1 = [v for e in eps for v in (e.get("v1") or [])[:SB.N_PLAYERS]]; pl2 = [v for e in eps for v in (e.get("v2") or [])[:SB.N_PLAYERS]]
    by1 = [v for e in eps for v in (e.get("v1") or [])[SB.N_PLAYERS:]]
    games = [(e, g) for e in eps for g in ("g1", "g2") if e.get(f"played{g[-1]}")]
    anc = [e for e in eps if e["o"]["g"]["source"] == "anchor"]; prop = [e for e in eps if e["o"]["g"]["source"] == "proposal"]
    sanc = defaultdict(list)
    for e, g in games:
        verbs = SN.player_verbs(e["o"]["g"]["rules"])
        if verbs:
            kind = "ctl" if SN.is_control_game(e["o"]["g"]["rules"]) else "eff"
            for s in range(SB.N_PLAYERS):
                if e[f"{g}_inst"]["seats"][s] == LLM and (e.get("types") or [PLAYER] * SB.N_PLAYERS)[s] == PLAYER:
                    st = SN.seat_stats(e[f"{g}_trace"], verbs, e.get("types") or [PLAYER] * SB.N_PLAYERS, s, SN.rule_of(e["o"]["g"]["rules"]))
                    if st["sanction_share"] is not None:
                        sanc[f"sanction_share_{kind}"].append(st["sanction_share"]); sanc[f"sanction_just_{kind}"].append(st["sanction_just"])
    terms = defaultdict(list)
    for e, g in games:
        for t in e[f"{g}_terms"]:
            if t:
                for k, v in t.items():
                    terms[f"fun_{k}"].append(v)
    return {"step": step, "run": spec["name"], "seconds": round(time.time() - ts, 1), **{f"t_{k}": round(v, 2) for k, v in tim.items()},
            "valid_share": sum(v["ok"] for v in verdicts) / max(1, P), "proposer_reward": m(p_rew), "gate": 0.0, "buffer": len(buf.items),
            "reasons": dict(Counter(v["reason"] for v in verdicts if not v["ok"])), "kl_proposer": kl_p / max(1, n_p),
            "play_share_v1": m([float(v in (VR.PLAY, VR.RESET)) for v in pl1]), "play_share_v2": m([float(v in (VR.PLAY, VR.RESET)) for v in pl2]),
            "reset_share": m([float(v == VR.RESET) for v in pl1 + pl2]), "bystander_allow": m([float(v == VR.PLAY) for v in by1]) if by1 else None,
            "played1": m([float(bool(e.get("played1"))) for e in eps]), "played2": m([float(bool(e.get("played2"))) for e in eps]),
            "pyrrhic": n_pyr / max(1, len(eps)), "game_errors": sum(bool(e.get("g1_err") or e.get("g2_err")) for e in eps),
            "anchor_play_share": m([float(v in (VR.PLAY, VR.RESET)) for e in anc for v in (e.get("v1") or [])[:SB.N_PLAYERS]]) if anc else None,
            "proposal_play_share": m([float(v in (VR.PLAY, VR.RESET)) for e in prop for v in (e.get("v1") or [])[:SB.N_PLAYERS]]) if prop else None,
            "bystander_loss": m([e[f"{g}_ev"]["bystander_loss"] for e, g in games]), "top_share": m([e[f"{g}_ev"]["top_share"] for e, g in games]),
            "stomps": m([float(e[f"{g}_ev"].get("stomp", 0)) for e, g in games]), "gain": m([x for e, g in games for x in e[f"{g}_gains"]]),
            "reward_player": m([x for e, g in games for s_, x in enumerate(e[f"{g}_rew"]) if x is not None and (e.get("types") or [PLAYER] * SB.N_PLAYERS)[s_] == PLAYER]),
            **{k: m(v) for k, v in terms.items()}, **{k: m(v) for k, v in sanc.items()},
            "decisions": len(D), "decisions_trained": sum(d["adv"] is not None for d in D), "vote_decisions": sum(d["key"][1] in ("v1", "v2") for d in D),
            **{f"grad_norm_{r}": v for r, v in gn_role.items()}, **{f"kl_{w.split(':')[-1]}": m(v) for w, v in kl_role.items()},
            "base_cache_hit": cnt["base_hit"], "base_cache_miss": cnt["base_miss"], "infer_batch": pol.infer_batch, "micro_batch": pol.micro_batch,
            "world_wallets": [list(wd["w"].wallets) for wd in worlds], "world_bystanders": [list(wd["w"].bystanders) for wd in worlds]}


@torch.no_grad()
def vote_eval(nets: A3.Nets, pol: PO.LetterPolicy, panel: Sequence[Dict], anchors: Sequence[Dict], seed: int, worlds_per_game: int) -> List[Dict]:
    """P(play) of the current player adapter and of the untrained model on every panel game and every anchor, from fresh worlds and
    from unequal ones (no sampling, no play): which games do the players want, and does wanting track what the game does?"""
    rng = random.Random(seed); meta, prompts = [], []
    games = [dict(g, kind="anchor") for g in anchors] + [dict(g, kind="control" if SN.is_control_game(g["rules"]) else
                                                              ("sanction" if SN.player_verbs(g["rules"]) else "seed")) for g in panel]
    for g in games:
        for k in range(worlds_per_game):
            w = EC.World(seed=seed + k)
            if k % 2:
                w.wallets = sorted(rng.randint(1, 25) for _ in range(SB.N_PLAYERS)); w.bystanders = [rng.randint(2, 10) for _ in range(SB.N_BYSTANDERS)]
            for s in range(SB.N_PLAYERS):
                prompts.append(vote_prompt(g["rules"], w, s, False, False, None, [PLAYER_OPTIONS[VR.PLAY], PLAYER_OPTIONS[VR.DECLINE]]))
                meta.append({"game": g.get("name"), "kind": g["kind"], "world": k, "seat": s, "unequal": bool(k % 2), "wallet": w.wallets[s],
                             "max_wallet": max(w.wallets)})
    rows = []
    whos = ["cur:" + PLAYER, "base"] + (["cur:" + WINNER] if nets.roles and WINNER in nets.roles else [])    # azr12: winners too
    for who in whos:
        with nets.use(who):
            probs = pol.probs(prompts, [2] * len(prompts), systems=[latch(WINNER if who.endswith(":" + WINNER) else PLAYER)] * len(prompts))
        rows += [dict(mm, who=who, p_play=p[0]) for mm, p in zip(meta, probs)]
    return rows
