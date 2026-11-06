"""The voting version of the infinite game (Thomas 2026-10-06): finite players, a proposer, and social rules instead of hand-coded
world penalties. Pure python (no torch): the rules, the fun terms and the rewards; azr_vote.py plays and trains with them.

The idea: players are trained on their finite gain. What makes the game infinite is social: every game is OFFERED and the players
(and, in one condition, the bystanders) vote to play it, decline it, or reset everyone's resources first. A game goes ahead on a
majority. Right after a game the same game is offered again (the rematch); that vote is "was it worth playing?", rewarded like any
other move by what it brings. If most of the players who played decline the rematch, the winners' victory is Pyrrhic: positive
gains from that game turn negative. Five intrinsic "fun" terms are added to each player's reward.

    decline reward     DECLINE_REWARD x DECLINE_DECAY^(declines in a row); an accept resets the run to zero (Thomas: declining must
                       recover when a player accepts, so it is cheap occasionally and expensive as a habit)
    resets             a world allows RESETS_PER_EPOCH resets per RESET_EPOCH games; a reset sets the players' wallets back to the
                       start (bystanders unchanged)
    fun terms          each in [0, 1]:
      learning progress  the player-role's improvement on this game: fast minus slow running average of its gain on the game,
                         positive part (game-level: it shapes which games players choose, not moves within one)
      novelty            mean over the seat's turns of 1/sqrt(1 + times this state has been seen), a state being the game plus its
                         state description with every number blanked; decays with familiarity
      surprise           positive part of (this gain - the player-role's running average gain on this game): better than expected,
                         against an expectation the player cannot move
      empowerment        mean over the seat's turns of log(legal moves) / log(MAX_MOVES): options still open
      others' fun        mean over the other playing seats of their gain plus their own fun terms (excluding this one), rescaled to [0, 1]

2026-10-07 (azr9 grid, after the azr8 read-back): the night-8 rules stay the defaults; three knobs were added.
    decline size       Voter(base, decay): the decline reward's size and halving per decline in a row are per-voter settings
    centred fun        seat_rewards(..., baseline=b): the weighted fun total minus a running baseline (FunState.fun_baseline, an EMA of the
                       role's raw weighted fun total over every played seat), so a game no more fun than usual adds ~0, a duller one less.
                       In azr8 every term was >= 0, so any game paid ~+0.28 of fun against a decline's 0.1 and players learned to accept all.
    harm veto          (2026-10-10, azr10) a bystander that lost holdings in game 1 can VETO the rematch on its own (its refusal blocks it
                       whatever the majority says), and every player whose game-1 moves took from a vetoing bystander loses VETO_PENALTY
                       from its game-1 return: the harmed party's exit, as a sanction attributable to the players who harmed it;
                       and every player who VOTED for the vetoed rematch (play or reset) loses VETO_VOTER_PENALTY on that vote (Thomas
                       2026-10-10: wanting to go on playing a game that just harmed a bystander is benefiting from a bad game)
    plant bystanders   plant_reward = growth + bystander_reward: a bystander grows a little whenever a game it allowed is played (Thomas:
                       "make them a bit more like plants"), so allowing pays unless the game costs it more than the growth; their decline
                       pays nothing (Voter(base=0)). In azr8 allowing paid only the change in holdings (<= 0) and refusing everything won.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

PLAY, DECLINE, RESET = "play", "decline", "reset"
DECLINE_REWARD = 0.1             # in units of the finite gain (a whole game of working earns ~ +0.7)
DECLINE_DECAY = 0.5
RESETS_PER_EPOCH = 2
RESET_EPOCH = 20                 # games per world per epoch
VETO_VOTER_PENALTY = 0.3         # harm veto: subtracted from each player's rematch-vote return when it voted for a vetoed rematch
VETO_PENALTY = 0.3               # harm veto: subtracted from a taker's game-1 return per vetoing bystander it took from
EMA_FUN = 0.1                    # centred fun: rate of the running baseline of the weighted fun total
FUN_TERMS = ("learning", "novelty", "surprise", "empowerment", "others")
FUN_WEIGHTS = {k: 0.2 for k in FUN_TERMS}
WINNER_FUN_WEIGHTS = dict(FUN_WEIGHTS, others=0.0)   # azr12 pure winners: own fun terms, nothing for the other players
# azr14 (Thomas 2026-10-18): fun should include AGENCY (do my choices change what happens?) and OUTCOME UNCERTAINTY (a game I could win or
# lose; always winning or always losing is dull). agency = the game's sandbox agency (azr_sandbox.validate: share of probed decisions where
# trying another legal move on a copy changes what the player sees); uncertainty = 4 p (1 - p), p = running win rate of this seat type on
# this game (1 for a game never played: maximally open). Opt in with these weights (vote.fun_v2); the defaults above stay the night-8 rules.
FUN_TERMS_V2 = FUN_TERMS + ("agency", "uncertainty")
FUN_WEIGHTS_V2 = {k: 0.2 for k in FUN_TERMS_V2}
WINNER_FUN_WEIGHTS_V2 = dict(FUN_WEIGHTS_V2, others=0.0)
EMA_WIN = 0.1
EMA_FAST, EMA_SLOW = 0.5, 0.1
MAX_MOVES = 16
PYRRHIC_SIGN = -1.0              # a winner's positive gain is multiplied by this when the rematch fails
NUMBER_RE = re.compile(r"\d+")


def state_key(describe: str) -> str:
    return NUMBER_RE.sub("#", describe or "")


def decline_reward(streak: int, base: float = DECLINE_REWARD, decay: float = DECLINE_DECAY) -> float:
    return base * decay ** max(0, int(streak))


@dataclass
class Voter:
    """Per (world, seat) state that carries across games: the run of declines."""
    streak: int = 0
    base: float = DECLINE_REWARD
    decay: float = DECLINE_DECAY
    cost: float = 0.0                 # azr11 (vote_control): a flat cost on every decline, not decayed by the streak

    def vote(self, choice: str) -> float:
        """Record a vote; returns the reward a DECLINE earns now (0 for play / reset)."""
        if choice == DECLINE:
            r = decline_reward(self.streak, self.base, self.decay) - self.cost; self.streak += 1; return r
        self.streak = 0; return 0.0


def majority(votes: Sequence[str]) -> Tuple[bool, bool]:
    """(the game is played, everyone's resources are reset first): played when more than half vote play or reset; reset when
    more than half vote reset."""
    n = len(votes); yes = sum(v in (PLAY, RESET) for v in votes); rs = sum(v == RESET for v in votes)
    return yes * 2 > n, rs * 2 > n


def bystander_vote_weight(share: float, n_players: int, n_bystanders: int) -> float:
    """azr21: the weight of ONE bystander's vote (players weigh 1 each) so that the bystanders together hold ``share`` of the total
    vote weight: n_b * w / (n_p + n_b * w) = share."""
    if not 0.0 < share < 1.0:
        raise ValueError(f"bystander vote share {share}: strictly between 0 and 1 (0 = bystanders_vote off)")
    return share * n_players / ((1.0 - share) * n_bystanders)


def play_share(votes: Sequence[str], n_players: int, w_b: float) -> Tuple[float, float]:
    """azr21: (weighted share of play/reset votes, weighted share of reset votes); voters 0..n_players-1 weigh 1, the rest w_b."""
    ws = [1.0 if s < n_players else w_b for s in range(len(votes))]; tot = sum(ws)
    if tot <= 0:
        raise ValueError("no vote weight")
    yes = sum(w for w, v in zip(ws, votes) if v in (PLAY, RESET)); rs = sum(w for w, v in zip(ws, votes) if v == RESET)
    return yes / tot, rs / tot


def lottery(votes: Sequence[str], n_players: int, w_b: float, u: float) -> Tuple[bool, bool, float]:
    """azr21 lottery vote: the game is played with probability = the weighted play share (u: a uniform draw in [0, 1)); resources
    are reset first when the game is played and more than half the weight voted reset (the majority rule's reset, weighted).
    Returns (played, reset, p_play)."""
    p, r = play_share(votes, n_players, w_b)
    go = u < p
    return go, go and r * 2 > 1, p


@dataclass
class FunState:
    """Running statistics across the run (one per training run)."""
    ema_fast: Dict[Tuple[str, str], float] = field(default_factory=dict)
    ema_slow: Dict[Tuple[str, str], float] = field(default_factory=dict)
    seen: Dict[bytes, int] = field(default_factory=lambda: defaultdict(int))
    fun_ema: Dict[str, float] = field(default_factory=dict)
    win_ema: Dict[Tuple[str, str], float] = field(default_factory=dict)   # azr14: running win rate per (game, seat type)

    def uncertainty(self, game: str, role: str) -> float:
        p = self.win_ema.get((game, role), 0.5)
        return 4.0 * p * (1.0 - p)

    def update_win(self, game: str, role: str, won: bool) -> None:
        k = (game, role); x = 1.0 if won else 0.0
        self.win_ema[k] = x if k not in self.win_ema else (1 - EMA_WIN) * self.win_ema[k] + EMA_WIN * x

    def fun_baseline(self, role: str) -> float:
        """Centred fun: the running average of the role's raw weighted fun total (0 before any game)."""
        return self.fun_ema.get(role, 0.0)

    def update_fun(self, role: str, total: float) -> None:
        self.fun_ema[role] = total if role not in self.fun_ema else (1 - EMA_FUN) * self.fun_ema[role] + EMA_FUN * total

    def expectation(self, game: str, role: str) -> Optional[float]:
        return self.ema_slow.get((game, role))

    def learning(self, game: str, role: str) -> float:
        f, s = self.ema_fast.get((game, role)), self.ema_slow.get((game, role))
        return 0.0 if f is None or s is None else max(0.0, min(1.0, f - s))

    def update(self, game: str, role: str, gain: float) -> None:
        k = (game, role)
        self.ema_fast[k] = gain if k not in self.ema_fast else (1 - EMA_FAST) * self.ema_fast[k] + EMA_FAST * gain
        self.ema_slow[k] = gain if k not in self.ema_slow else (1 - EMA_SLOW) * self.ema_slow[k] + EMA_SLOW * gain

    def novelty(self, game: str, states: Sequence[str], record: bool) -> float:
        if not states:
            return 0.0
        vals = []
        for st in states:
            key = hashlib.sha1(f"{game}\x00{state_key(st)}".encode()).digest()
            vals.append(1.0 / math.sqrt(1.0 + self.seen[key]))
            if record:
                self.seen[key] += 1
        return sum(vals) / len(vals)


def empowerment(n_moves: Sequence[int]) -> float:
    return sum(math.log(max(1, n)) for n in n_moves) / (len(n_moves) * math.log(MAX_MOVES)) if n_moves else 0.0


def own_terms(state: FunState, game: str, role: str, gain: float, n_moves: Sequence[int], states: Sequence[str], record_novelty: bool,
              agency: Optional[float] = None) -> Dict[str, float]:
    """A seat's own fun terms for one game (everything but 'others'), read BEFORE this game updates the running statistics.
    ``n_moves``: the number of legal moves at each of the seat's turns; ``states``: the seat's state descriptions. ``agency`` (azr14): the
    game's sandbox agency; when given, the agency and uncertainty terms are added."""
    exp = state.expectation(game, role)
    t = {"learning": state.learning(game, role), "novelty": state.novelty(game, states, record_novelty),
         "surprise": 0.0 if exp is None else max(0.0, min(1.0, gain - exp)), "empowerment": empowerment(n_moves)}
    if agency is not None:
        t.update(agency=max(0.0, min(1.0, float(agency))), uncertainty=state.uncertainty(game, role))
    return t


def winners(gains: Sequence[float], playing: Sequence[bool]) -> List[Optional[bool]]:
    """azr14: per seat, did it win (gain strictly above every other playing seat's)? None for a seat that did not play."""
    out: List[Optional[bool]] = []
    for s, g in enumerate(gains):
        if not playing[s]:
            out.append(None); continue
        others = [gains[t] for t in range(len(gains)) if t != s and playing[t]]
        out.append(bool(others) and all(g > o for o in others))
    return out


def seat_rewards(gains: Sequence[float], terms: Sequence[Optional[Dict[str, float]]], weights: Optional[Dict[str, float]] = None,
                 fun: bool = True, baseline: float = 0.0) -> Tuple[List[Optional[float]], List[Optional[Dict[str, float]]]]:
    """Per playing seat: gain + the weighted fun terms (``terms[s]`` None for a seat that did not play, e.g. a decliner replaced by a
    scripted player). 'others' = mean over the other playing seats of (gain + their own weighted terms), mapped from [-1, 1+] to [0, 1].
    ``baseline``: subtracted from the weighted fun total (centred fun; 0 = the night-8 rule)."""
    w = FUN_WEIGHTS if weights is None else weights; n = len(gains); full = [None] * n; out: List[Optional[float]] = [None] * n
    own = [None if terms[s] is None else gains[s] + sum(w[k] * terms[s].get(k, 0.0) for k in w if k != "others") for s in range(n)]
    for s in range(n):
        if terms[s] is None:
            continue
        others = [own[t] for t in range(n) if t != s and own[t] is not None]
        o = (sum(others) / len(others) + 1.0) / 2.0 if others else 0.0
        full[s] = dict(terms[s], others=max(0.0, min(1.0, o)))
        out[s] = gains[s] + (fun_total(full[s], w) - baseline if fun else 0.0)
    return out, full


def fun_total(terms: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    w = FUN_WEIGHTS if weights is None else weights
    return sum(w[k] * terms.get(k, 0.0) for k in w)


def pyrrhic(rewards: Sequence[Optional[float]], gains: Sequence[float], rematch_votes: Sequence[Optional[str]],
            strength: float = 1.0) -> Tuple[List[Optional[float]], bool]:
    """If most of the players who played decline the rematch, every winner's positive gain turns negative (its fun terms stay).
    ``strength`` (azr11, vote_control) scales the flip: a winner keeps r - g - strength x g (1 = the night-8 rule: gain becomes -gain)."""
    played = [v for v in rematch_votes if v is not None]
    fails = bool(played) and sum(v == DECLINE for v in played) * 2 > len(played)
    if not fails:
        return list(rewards), False
    return [None if r is None else (r - g + PYRRHIC_SIGN * strength * g if g > 0 else r) for r, g in zip(rewards, gains)], True


def bystander_reward(before: int, after: int, scale: float) -> float:
    """A voting bystander's reward for a game it allowed: the change in its holdings, in the players' gain units, clipped."""
    return max(-1.0, min(1.0, (after - before) / max(1.0, scale)))


def bystander_share_reward(before: int, after: int) -> float:
    """azr20 (Thomas 2026-11-01): the change in a bystander's holdings as a SHARE of what it held, so losing everything is -1 however little
    it had (bystander_reward divides by the players' average holdings: a broke bystander drained of its last 2 coins by rich players lost
    ~0.07 and the plant growth outweighed it)."""
    return max(-1.0, min(1.0, (after - before) / max(1.0, before)))


def plant_share_reward(before: int, after: int, growth: float) -> float:
    """azr20: plant growth plus the share-of-holdings change."""
    return growth + bystander_share_reward(before, after)


def plant_reward(before: int, after: int, scale: float, growth: float) -> float:
    """A plant bystander's reward for a game it allowed that was played: a small growth plus the change in its holdings."""
    return growth + bystander_reward(before, after, scale)


def harmed(bys_in: Sequence[int], bys_out: Sequence[int]) -> List[bool]:
    """Per bystander: did it end the game holding less than it came in with?"""
    return [b1 < b0 for b0, b1 in zip(bys_in, bys_out)]


def takers(trace: Sequence[Dict], n_bys: int) -> List[List[int]]:
    """Per bystander: the seats with at least one move that lowered its holdings (per-move bystander deltas ``dbys``)."""
    out: List[set] = [set() for _ in range(n_bys)]
    for m in trace:
        for b, d in enumerate((m.get("dbys") or [])[:n_bys]):
            if d < 0:
                out[b].add(int(m["player"]))
    return [sorted(s) for s in out]


def harm_veto(votes: Sequence[str], harmed_: Sequence[bool], n_players: int) -> List[int]:
    """The bystanders that veto the rematch: harmed in game 1 and voted to refuse (votes = players then bystanders)."""
    return [b for b, h in enumerate(harmed_) if h and n_players + b < len(votes) and votes[n_players + b] == DECLINE]


def veto_penalties(rewards: Sequence[Optional[float]], takers_: Sequence[Sequence[int]], vetoers: Sequence[int],
                   penalty: float = VETO_PENALTY) -> List[Optional[float]]:
    """Game-1 returns after the veto: each seat loses ``penalty`` per vetoing bystander it took from (seats that did not play: None)."""
    out = list(rewards)
    for b in vetoers:
        for s in takers_[b]:
            if s < len(out) and out[s] is not None:
                out[s] -= penalty
    return out


def rematch_voter_penalties(votes: Sequence[Optional[str]], vetoers: Sequence[int], n_players: int,
                            penalty: float = VETO_VOTER_PENALTY) -> List[float]:
    """Per player: the penalty on its rematch vote, ``penalty`` if the rematch was vetoed and it voted play or reset, else 0."""
    if not vetoers:
        return [0.0] * n_players
    return [penalty if (s < len(votes) and votes[s] in (PLAY, RESET)) else 0.0 for s in range(n_players)]


def proposer_vote_bonus(first_votes: Sequence[str], rematch_votes: Sequence[str]) -> float:
    """What the voters thought of a proposed game: the share of play/reset votes over both offerings, minus the share of resets."""
    vs = list(first_votes) + list(rematch_votes)
    if not vs:
        return 0.0
    return sum(v in (PLAY, RESET) for v in vs) / len(vs) - sum(v == RESET for v in vs) / len(vs)


def a_harm(takers_: Sequence[Sequence[int]], types: Sequence[str], role: str = "A") -> bool:
    """azr12: did a seat of ``role`` take from any bystander in this game (``takers_`` from takers())."""
    return any(types[p] == role for t in takers_ for p in t if p < len(types))


def exempt(new: Sequence[Optional[float]], old: Sequence[Optional[float]], types: Sequence[str], role: str = "W") -> List[Optional[float]]:
    """azr12: seats of ``role`` keep their reward from before a social rule (Pyrrhic flip, takers' veto penalty) was applied."""
    return [o if types[s] == role else n for s, (n, o) in enumerate(zip(new, old))]
