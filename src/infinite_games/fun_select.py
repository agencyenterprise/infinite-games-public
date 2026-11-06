"""azr19 (Thomas 2026-10-26): pick buffer games by how much fun they were, not only by validity.

Until azr18 every valid game joined the buffer and was then drawn uniformly, both as the proposer's reference examples and as buffer
offerings. azr17's proposers settled on their game style within ~30 updates and never moved (infinite-only games strict agency ~.33,
finite-only ~.06) while the proposer's vote reward did not track agency; the seed games stayed far more fun than any proposer game.

Here each game's FUN SCORE is a running average (EMA) of the focus players' raw weighted fun total over its played game 1s, with the
type's base weights (not the controller's fun scale, which multiplies every game alike and would only add drift). A draw is uniform
with probability ``share_uniform`` (Thomas: half stays random); otherwise rank-weighted on the fun score (weight = rank ** power, rank 1
= least fun), a game never played counting as the median score. Torch-free.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

EMA = 0.3                     # weight of a new game-1 fun total in a game's score
SHARE_UNIFORM = 0.5
POWER = 1.0


class FunScores:
    def __init__(self, ema: float = EMA):
        self.ema = float(ema); self.score: Dict[str, float] = {}; self.plays: Dict[str, int] = {}

    def update(self, name: str, fun: float) -> None:
        old = self.score.get(name)
        self.score[name] = float(fun) if old is None else (1 - self.ema) * old + self.ema * float(fun)
        self.plays[name] = self.plays.get(name, 0) + 1

    def get(self, name: str) -> Optional[float]:
        return self.score.get(name)


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs); n = len(s)
    return 0.0 if not n else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def weights(pool: Sequence[Dict], scores: FunScores, power: float = POWER) -> List[float]:
    """Rank weights over ``pool`` (ties share their mean rank); unplayed games take the median of the scored ones."""
    known = [scores.get(it.get("name", "")) for it in pool]
    med = _median([x for x in known if x is not None])
    vals = [med if x is None else x for x in known]
    order = sorted(range(len(vals)), key=lambda i: vals[i]); rank = [0.0] * len(vals); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rank[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return [r ** float(power) for r in rank]


def pick(pool: Sequence[Dict], rng: random.Random, scores: FunScores, share_uniform: float = SHARE_UNIFORM, power: float = POWER) -> Dict:
    if rng.random() < float(share_uniform) or not scores.score:
        return pool[rng.randrange(len(pool))]
    return rng.choices(list(pool), weights=weights(pool, scores, power), k=1)[0]
