"""azr15 (Thomas 2026-10-19): players PREDICT their own reward before they vote, part by part, and see what they last experienced.

azr14 read-back: the voters' verdict predicted nothing (r 0.03 with the realised game reward, 0.04 with agency, n 1386), because every game
was new, the vote saw only the rules text, and the vote was learned only through one play/decline bit. Thomas: train the prediction the
easy way (supervised), split it into the reward's parts, show the current mix, and let the model build the total from its part predictions.

    parts        COMPONENTS: the finite gain, the social rules' net effect (Pyrrhic flip, veto penalties), and each fun term. TOTAL is the
                 reward the seat actually received for game 1.
    bands        each part is predicted as one of N_BINS bands. Band edges are running quintiles of the realised values of that part for
                 that seat type (Binner: the last ``window`` values); a part is predicted only once ``min_n`` values exist. The target of a
                 prediction is binned with the edges the prediction was made with (snapshot).
    total        a second prompt shows the current mix (gain x 1 + social x 1 + fun weight x term - fun baseline) and the seat's own part
                 predictions (most likely band each), and asks for the total's band. The vote prompt then carries that predicted band.
    memory       what the seat saw the last time it played this game (a trial play for a new game, else its last real game 1): its moves,
                 its holdings before/after, the bystanders' before/after, who won. Facts only; no sandbox scores.
    fun share    the mixer statistic: among type-A seats' played game 1s, var(fun part) / (var(fun part) + var(gain + social)).
Pure python."""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from . import vote_rules as VR

N_BINS = 5
QUANTILES = (0.2, 0.4, 0.6, 0.8)
TOTAL = "total"
COMPONENTS = ("gain", "social") + VR.FUN_TERMS_V2
WINDOW = 1000
MIN_N = 50
MEMORY_MOVES = 8
MEMORY_CHARS = 700
BAND_LETTERS = "ABCDE"
PART_TEXT = {
    "gain": "your finite gain in this game (what you end with minus what you brought in, scaled)",
    "social": "what the rematch rules add to or take from you after this game (Pyrrhic flip, veto penalties; usually 0)",
    "learning": "your 'learning' fun term (how much faster you are improving at this game than usual)",
    "novelty": "your 'novelty' fun term (how unfamiliar the states of this game are to you)",
    "surprise": "your 'surprise' fun term (how much better than you usually do on this game)",
    "empowerment": "your 'empowerment' fun term (how many options stay open on your turns)",
    "others": "your 'others' fun term (how well the other players do)",
    "agency": "your 'agency' fun term (whether your choices change what happens)",
    "uncertainty": "your 'uncertainty' fun term (whether this game could go either way for players like you)",
    TOTAL: "your total reward for this game",
}


class Binner:
    """Running band edges per (seat type, part): quintiles of the last ``window`` realised values."""

    def __init__(self, window: int = WINDOW, min_n: int = MIN_N):
        self.window, self.min_n = int(window), int(min_n)
        self.vals: Dict[Tuple[str, str], Deque[float]] = {}

    def add(self, role: str, part: str, value: float) -> None:
        self.vals.setdefault((role, part), deque(maxlen=self.window)).append(float(value))

    def ready(self, role: str, part: str = TOTAL) -> bool:
        return len(self.vals.get((role, part), ())) >= self.min_n

    def edges(self, role: str, part: str) -> Optional[List[float]]:
        v = sorted(self.vals.get((role, part), ()))
        if len(v) < self.min_n:
            return None
        return [v[min(len(v) - 1, int(q * len(v)))] for q in QUANTILES]

    def centres(self, role: str, part: str, edges: Sequence[float]) -> List[float]:
        """Per band: the mean of the stored values that fall in it (edge midpoint for an empty band)."""
        acc: List[List[float]] = [[] for _ in range(N_BINS)]
        for x in self.vals.get((role, part), ()):
            acc[band(x, edges)].append(x)
        lo = [edges[0] - (edges[-1] - edges[0] or 1.0) / 4] + list(edges); hi = list(edges) + [edges[-1] + (edges[-1] - edges[0] or 1.0) / 4]
        return [sum(a) / len(a) if a else (lo[b] + hi[b]) / 2 for b, a in enumerate(acc)]


def band(x: float, edges: Sequence[float]) -> int:
    """0..N_BINS-1; a value equal to an edge goes to the lower band (so a constant part always lands in band 0)."""
    return bisect_left(list(edges), float(x))


def band_text(edges: Sequence[float]) -> List[str]:
    e = [f"{x:+.2f}" for x in edges]
    return [f"below {e[0]}"] + [f"{e[i]} to {e[i + 1]}" for i in range(len(e) - 1)] + [f"above {e[-1]}"]


def bands_block(edges: Sequence[float]) -> str:
    return "\n".join(f"{BAND_LETTERS[i]}) {t}" for i, t in enumerate(band_text(edges)))


def part_prompt(head: str, part: str, role: str, edges: Sequence[float]) -> str:
    return (f"{head}\n\nBefore you vote, predict {PART_TEXT[part]}, if you play it. Recent values of it for type {role} players "
            f"fall into five bands:\n{bands_block(edges)}\n\nReply with the letter of the band you expect.")


def mix_text(weights: Dict[str, float], baseline: float) -> str:
    fun = " + ".join(f"{w:.2f} x {k}" for k, w in weights.items() if w)
    return f"total = gain + social + ({fun or '0'}) - {baseline:.2f} (the running average of the fun part)"


def total_prompt(head: str, role: str, weights: Dict[str, float], baseline: float, part_bands: Dict[str, Tuple[int, List[float]]],
                 edges: Sequence[float]) -> str:
    lines = [f"{p}: band {BAND_LETTERS[b]} ({band_text(e)[b]})" for p, (b, e) in part_bands.items()]
    return (f"{head}\n\nYour reward for a game is built from parts. The current mix: {mix_text(weights, baseline)}.\n"
            f"Your own predictions of the parts:\n" + "\n".join(lines) +
            f"\n\nNow predict {PART_TEXT[TOTAL]}, if you play it. Recent totals for type {role} players fall into five bands:\n"
            f"{bands_block(edges)}\n\nReply with the letter of the band you expect.")


def vote_note(total_band: int, edges: Sequence[float], decline_value: float) -> str:
    return (f"Your prediction for this game: total reward band {BAND_LETTERS[total_band]} ({band_text(edges)[total_band]}). "
            f"Declining now pays you {decline_value:+.2f}.")


def expected(probs: Sequence[float], centres: Sequence[float]) -> float:
    return float(sum(p * c for p, c in zip(probs, centres)))


def composed(part_values: Dict[str, float], weights: Dict[str, float], baseline: float) -> float:
    """The total implied by part values under the mix (for comparison with the model's own total prediction)."""
    return part_values.get("gain", 0.0) + part_values.get("social", 0.0) + sum(w * part_values.get(k, 0.0) for k, w in weights.items()) - baseline


def parts_of(gain: float, social: float, terms: Optional[Dict[str, float]]) -> Dict[str, float]:
    out = {"gain": float(gain), "social": float(social)}
    for k in VR.FUN_TERMS_V2:
        out[k] = float((terms or {}).get(k, 0.0))
    return out


def memory_note(seat: int, moves: Sequence[str], before: int, after: int, bys_before: Sequence[int], bys_after: Sequence[int],
                won: Optional[bool], trial: bool, players_before: Optional[Sequence[int]] = None,
                players_after: Optional[Sequence[int]] = None) -> str:
    """``players_before`` / ``players_after`` (azr19, Thomas 2026-10-26): every player's holdings, so the note also says what the other
    players gained (the 'others' fun term pays for it and was absent from the prompt); None = the azr15 note unchanged."""
    mv = list(moves)[:MEMORY_MOVES]
    more = f" (+{len(moves) - len(mv)} more)" if len(moves) > len(mv) else ""
    res = "you won" if won else ("nobody won outright or another player came out ahead" if won is not None else "")
    co = ""
    if players_before is not None and players_after is not None:
        co = "; the other players' holdings went " + ", ".join(f"P{i} {b} -> {a}" for i, (b, a) in enumerate(zip(players_before, players_after))
                                                              if i != seat)
    txt = (f"{'In a practice play' if trial else 'The last time you played'} of this game (as P{seat}): your holdings went {before} -> {after}"
           f"{co}; bystanders {list(bys_before)} -> {list(bys_after)}" + (f"; {res}" if res else "")
           + f"; your moves were {', '.join(mv) or 'none'}{more}.") if co else (
           f"{'In a practice play' if trial else 'The last time you played'} of this game (as P{seat}): your moves were "
           f"{', '.join(mv) or 'none'}{more}; your holdings went {before} -> {after}; bystanders {list(bys_before)} -> {list(bys_after)}"
           + (f"; {res}." if res else "."))
    return txt[:MEMORY_CHARS]


def fun_share(rows: Sequence[Tuple[float, float]]) -> Optional[float]:
    """``rows``: (fun part, gain + social) per type-A seat-game. var(fun) / (var(fun) + var(rest)); None below 3 rows or no variance."""
    if len(rows) < 3:
        return None
    var = lambda xs: sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)
    vf, vr = var([r[0] for r in rows]), var([r[1] for r in rows])
    return None if vf + vr <= 0 else vf / (vf + vr)


def scaled(weights: Dict[str, float], scale: float) -> Dict[str, float]:
    return {k: v * float(scale) for k, v in weights.items()}


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None

    def ranks(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    a, b = ranks(list(x)), ranks(list(y)); ma, mb = sum(a) / n, sum(b) / n
    sa = sum((u - ma) ** 2 for u in a) ** 0.5; sb = sum((u - mb) ** 2 for u in b) ** 0.5
    return None if sa == 0 or sb == 0 else sum((u - ma) * (w - mb) for u, w in zip(a, b)) / (sa * sb)
