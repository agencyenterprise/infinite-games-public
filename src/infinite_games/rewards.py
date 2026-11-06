"""Arm rewards, the matched-penalty calibration, group-relative advantages and bootstrap intervals. Pure python.

Arms (the reward an episode's focal player receives):
    win                 win share (1 / number tied at the top, else 0)                        -- the finite game
    win_minus_ko        win - lam * (players the focal knocked out)                           -- Thomas's penalty arm
    win_plus_options    win + mu * (mean remaining capacity of the other players at the end)   -- keep others able to play
    win_minus_invest    win - lam_d * (invest uses)                                            -- matched-penalty control:
                        a penalty on a harmless, self-regarding action, with lam_d set so the BASE policy would pay the
                        same expected penalty per episode as it pays under win_minus_ko
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

REWARDS = ("win", "win_minus_ko", "win_plus_options", "win_minus_invest")
LAM_D_BOUNDS = (0.05, 2.0)
BOOTSTRAP_DRAWS = 2000
CI_LEVEL = 0.95


def arm_reward(outcome: Dict, reward: str, lam: float = 1.0, mu: float = 0.5, lam_d: Optional[float] = None) -> float:
    w = float(outcome["win"])
    if reward == "win":
        return w
    if reward == "win_minus_ko":
        return w - lam * float(outcome["ko"])
    if reward == "win_plus_options":
        return w + mu * float(outcome["others_capacity"])
    if reward == "win_minus_invest":
        if lam_d is None:
            raise ValueError("win_minus_invest needs lam_d (from calibration)")
        return w - lam_d * float(outcome["invest"])
    raise ValueError(f"unknown reward {reward!r} (valid: {REWARDS})")


def matched_lambda(outcomes: Sequence[Dict], lam: float) -> Dict:
    """lam_d such that E_base[lam_d * invest] = E_base[lam * ko], clipped to LAM_D_BOUNDS. Returns the rates too."""
    if not outcomes:
        raise ValueError("no calibration outcomes")
    ko = sum(o["ko"] for o in outcomes) / len(outcomes); inv = sum(o["invest"] for o in outcomes) / len(outcomes)
    raw = (lam * ko / inv) if inv > 0 else LAM_D_BOUNDS[1]
    return {"lam_d": min(max(raw, LAM_D_BOUNDS[0]), LAM_D_BOUNDS[1]), "lam_d_raw": raw, "ko_rate": ko, "invest_rate": inv,
            "clipped": not (LAM_D_BOUNDS[0] <= raw <= LAM_D_BOUNDS[1])}


def group_advantages(rewards: Sequence[float], group: int) -> List[float]:
    """Reward minus its group's mean (no division by the group's spread: a near-constant group must not be amplified).
    Consecutive blocks of ``group`` rewards share a game."""
    if group <= 0 or len(rewards) % group:
        raise ValueError(f"{len(rewards)} rewards do not split into groups of {group}")
    out: List[float] = []
    for g in range(0, len(rewards), group):
        blk = rewards[g:g + group]; m = sum(blk) / group; out += [r - m for r in blk]
    return out


def mean_ci(xs: Sequence[float], seed: int = 0, draws: int = BOOTSTRAP_DRAWS, level: float = CI_LEVEL) -> Dict:
    """Mean with a percentile bootstrap interval (the mean is linear, so the interval contains the point estimate)."""
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return {"mean": m, "lo": m, "hi": m, "n": 1}
    rng = random.Random(seed); n = len(xs)
    boots = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    a = (1 - level) / 2
    lo, hi = boots[int(a * (draws - 1))], boots[int((1 - a) * (draws - 1))]
    return {"mean": m, "lo": min(lo, m), "hi": max(hi, m), "n": n}


def diff_ci(a: Sequence[float], b: Sequence[float], seed: int = 0, paired: bool = False, draws: int = BOOTSTRAP_DRAWS,
            level: float = CI_LEVEL) -> Dict:
    """mean(a) - mean(b) with a bootstrap interval; ``paired`` resamples index pairs (same items under two policies)."""
    a = [float(x) for x in a]; b = [float(x) for x in b]
    if not a or not b:
        return {"diff": None, "lo": None, "hi": None}
    if paired:
        if len(a) != len(b):
            raise ValueError("paired difference needs equal lengths")
        r = mean_ci([x - y for x, y in zip(a, b)], seed=seed, draws=draws, level=level)
        return {"diff": r["mean"], "lo": r["lo"], "hi": r["hi"]}
    d = sum(a) / len(a) - sum(b) / len(b); rng = random.Random(seed)
    boots = sorted(sum(a[rng.randrange(len(a))] for _ in a) / len(a) - sum(b[rng.randrange(len(b))] for _ in b) / len(b)
                   for _ in range(draws))
    q = (1 - level) / 2
    return {"diff": d, "lo": min(boots[int(q * (draws - 1))], d), "hi": max(boots[int((1 - q) * (draws - 1))], d)}


def excludes_zero(ci: Dict) -> Optional[bool]:
    if ci.get("lo") is None:
        return None
    return ci["lo"] > 0 or ci["hi"] < 0


def step_seeds(base_seed: int, step: int) -> Tuple[int, int]:
    """Game-sampling and sampling-noise seeds for a training step: identical for every arm (common random numbers)."""
    return (base_seed * 1_000_003 + step * 7919) % (2 ** 31), (base_seed * 999_983 + step * 104_729 + 1) % (2 ** 31)
