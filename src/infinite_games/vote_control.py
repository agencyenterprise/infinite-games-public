"""Dynamic reward weights for the voting game (azr11; Thomas 2026-10-12): aim at target STATISTICS, not at fixed hyperparameters.

azr10 fixed every weight and collapsed: playing drifted to a losing bet while harm stayed high, and after ~180 updates the players
declined everything. Here each protected statistic has a one-sided constraint, and a PID Lagrange multiplier (Stooke, Achiam & Abbeel
2020, arXiv 2007.03964: the classic multiplier update is integral control; P and D terms damp its oscillation and overshoot) rescales
the scores that protect it. Integral-only gains (kp = kd = 0) give the classic Lagrangian / dual-ascent update (RCPO, Tessler et al.
2018, arXiv 1805.11074), the ablation.

Constraints (statistics are shares in [0, 1], smoothed by an EMA over updates; every replay of the update counts):
    exit     decline share <= DECLINE_HI     lambda_exit:  declining starts to COST (decline reward - lambda_exit) and the group
                                                           punishments (Pyrrhic flip, voter penalty) are damped by 1 / (1 + lambda_exit)
    stay     decline share >= DECLINE_LO     lambda_stay:  the decline reward rises by lambda_stay and group punishments grow x (1 + lambda_stay)
    harm     harmful game-1 share <= HARM_HI mu_harm:      every harm penalty grows x (1 + mu_harm): the takers' veto penalty (never damped
                                                           by exit: whoever harms always pays), the voter penalty and the Pyrrhic flip
    byst     bystander allow share >= ALLOW_LO  lambda_byst: plant growth x (1 + lambda_byst)
    (azr18, Thomas 2026-10-26: no explicit theft rules) Controllers(harm=False) drops the harm constraint: no mu_harm, the takers'
             penalty and the group punishments never grow with the harmful-game share.
    fun      (azr15, opt-in: Controllers(fun_share_lo=x)) fun share >= x   lambda_fun: every type-A fun weight x (1 + lambda_fun). The fun
             share is var(fun part) / (var(fun part) + var(gain + social)) over type-A seats' played game 1s (vote_predict.fun_share);
             azr14 ran at ~0.1, so fun barely moved a seat's reward next to its gain (Thomas 2026-10-19: fun was a garnish).

Knobs (``Controllers.knobs``): decline_bonus (added to the decline reward base), decline_cost (subtracted from every decline, not decayed
by the streak), group_scale (Pyrrhic strength and voter penalty), taker_scale, growth_scale, fun_scale (1 without the fun
constraint). Pure python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

DECLINE_LO, DECLINE_HI = 0.10, 0.30
HARM_HI = 0.25
ALLOW_LO = 0.30
EMA = 0.2                       # weight of the newest update's statistic in the smoothed value
LAMBDA_MAX = 3.0                # multipliers are clipped to [0, LAMBDA_MAX]
PID = {"kp": 1.0, "ki": 0.1, "kd": 2.0}
INTEGRAL_ONLY = {"kp": 0.0, "ki": 0.1, "kd": 0.0}
CONSTRAINTS = {                  # name: (statistic key, threshold, sense: +1 = upper bound (stat <= thr), -1 = lower bound (stat >= thr))
    "exit": ("decline", DECLINE_HI, +1),
    "stay": ("decline", DECLINE_LO, -1),
    "harm": ("harm", HARM_HI, +1),
    "byst": ("allow", ALLOW_LO, -1),
}
FUN_CONSTRAINT = "fun"
FUN_STAT = "fun_share"


@dataclass
class PIDLagrangian:
    """One multiplier for one one-sided constraint, error e = sense x (stat - threshold) (positive = violated).
    lambda = clip(kp e + I + kd max(0, e - e_prev), 0, LAMBDA_MAX), I = clip(I + ki e, 0, LAMBDA_MAX) (anti-windup: the integral never
    goes negative and never beyond the cap). The derivative acts only on a worsening error (Stooke et al.'s damping).
    ``max_step`` (azr16, Thomas 2026-10-20): the multiplier moves at most this much per update, up or down (None = the azr11-15 rule). While
    the step limit binds, the integral is frozen (conditional integration), so it cannot wind up behind the limit. azr15: the exit cost
    rose 0.3 -> 1.4 in ~20 updates on the derivative + integral, then took ~35 updates to unwind, and the players swung from 85% declines
    to none."""
    kp: float
    ki: float
    kd: float
    threshold: float
    sense: int
    integral: float = 0.0
    prev: Optional[float] = None
    value: float = 0.0
    max_step: Optional[float] = None

    def update(self, stat: Optional[float]) -> float:
        if stat is None:                       # nothing measured this update (e.g. no game played): hold
            return self.value
        e = self.sense * (stat - self.threshold)
        integral = min(LAMBDA_MAX, max(0.0, self.integral + self.ki * e))
        d = max(0.0, e - self.prev) if self.prev is not None else 0.0
        self.prev = e
        raw = min(LAMBDA_MAX, max(0.0, self.kp * e + integral + self.kd * d))
        if self.max_step is not None and abs(raw - self.value) > self.max_step:
            self.value = self.value + (self.max_step if raw > self.value else -self.max_step)   # limited; integral frozen
        else:
            self.integral = integral; self.value = raw
        return self.value


@dataclass
class Controllers:
    gains: Dict[str, float]
    ema: float = EMA
    smoothed: Dict[str, Optional[float]] = field(default_factory=dict)
    pid: Dict[str, PIDLagrangian] = field(default_factory=dict)
    fun_share_lo: Optional[float] = None          # azr15: lower bound on the fun share (None = no fun constraint, the azr11-14 rules)
    cons: Dict[str, tuple] = field(default_factory=dict)
    max_step: Optional[float] = None              # azr16: per-update limit on every multiplier's change
    harm: bool = True                             # azr18: False = no harm constraint (no controller aimed at theft)

    def __post_init__(self):
        self.cons = dict(CONSTRAINTS)
        if not self.harm:
            del self.cons["harm"]
        if self.fun_share_lo is not None:
            self.cons[FUN_CONSTRAINT] = (FUN_STAT, float(self.fun_share_lo), -1)
        for n, (_, thr, sense) in self.cons.items():
            self.pid[n] = PIDLagrangian(float(self.gains["kp"]), float(self.gains["ki"]), float(self.gains["kd"]), thr, sense,
                                        max_step=None if self.max_step is None else float(self.max_step))

    def update(self, stats: Dict[str, Optional[float]]) -> Dict[str, float]:
        """``stats``: this update's raw decline / harm / allow shares (None = not measured). Returns the multipliers."""
        for k, v in stats.items():
            if v is not None:
                old = self.smoothed.get(k)
                self.smoothed[k] = v if old is None else (1 - self.ema) * old + self.ema * v
        return {n: c.update(self.smoothed.get(self.cons[n][0])) for n, c in self.pid.items()}

    def lambdas(self) -> Dict[str, float]:
        return {n: c.value for n, c in self.pid.items()}

    def knobs(self) -> Dict[str, float]:
        L = self.lambdas()
        return {"decline_bonus": L["stay"], "decline_cost": L["exit"],
                "group_scale": (1 + L.get("harm", 0.0)) * (1 + L["stay"]) / (1 + L["exit"]),
                "taker_scale": 1 + L.get("harm", 0.0), "growth_scale": 1 + L["byst"], "fun_scale": 1 + L.get(FUN_CONSTRAINT, 0.0)}


def fixed_knobs() -> Dict[str, float]:
    """The knobs of a run without controllers (azr10's rules unchanged)."""
    return {"decline_bonus": 0.0, "decline_cost": 0.0, "group_scale": 1.0, "taker_scale": 1.0, "growth_scale": 1.0, "fun_scale": 1.0}
