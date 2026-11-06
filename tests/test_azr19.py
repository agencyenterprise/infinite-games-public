"""azr19 (Thomas 2026-10-26): azr18 plus the other players' holdings in the prediction / first-vote memory note. Torch-free."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _load  # noqa: E402

VP = _load("vote_predict")
FS = _load("fun_select")
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"


def _cfg(name):
    return yaml.safe_load(open(CFG / f"{name}.yaml"))


def test_memory_note_unchanged_without_players():
    m = VP.memory_note(1, ["take 2"], 10, 14, [10, 10], [8, 10], True, trial=True)
    assert m == VP.memory_note(1, ["take 2"], 10, 14, [10, 10], [8, 10], True, trial=True, players_before=None, players_after=None)
    assert "other players" not in m and m.startswith("In a practice play")


def test_memory_note_names_the_other_players():
    m = VP.memory_note(1, [f"move{i}" for i in range(30)], 10, 14, [10, 10], [8, 10], False, trial=False,
                       players_before=[9, 10, 11], players_after=[12, 14, 5])
    assert "the other players' holdings went P0 9 -> 12, P2 11 -> 5" in m
    assert "P1 10 -> 14" not in m and "your holdings went 10 -> 14" in m
    assert len(m) <= VP.MEMORY_CHARS




def test_fun_scores_ema():
    f = FS.FunScores(ema=0.5)
    f.update("g", 1.0); f.update("g", 0.0)
    assert f.get("g") == 0.5 and f.plays["g"] == 2 and f.get("h") is None


def test_rank_weights_unplayed_take_median_and_ties_share():
    f = FS.FunScores()
    for n, x in (("a", 0.1), ("b", 0.5), ("c", 0.9)):
        f.update(n, x)
    w = FS.weights([{"name": n} for n in ("a", "b", "c", "new")], f)
    assert w[0] == 1.0 and w[3] == w[1] == 2.5 and w[2] == 4.0


def test_pick_is_uniform_without_scores_and_favours_fun():
    import random
    pool = [{"name": "dull"}, {"name": "fun"}]
    f = FS.FunScores()
    rng = random.Random(0)
    assert {FS.pick(pool, rng, f, share_uniform=0.0)["name"] for _ in range(200)} == {"dull", "fun"}
    f.update("dull", 0.0); f.update("fun", 1.0)
    picks = [FS.pick(pool, rng, f, share_uniform=0.0, power=4.0)["name"] for _ in range(2000)]
    assert picks.count("fun") > 0.9 * len(picks)
    half = [FS.pick(pool, rng, f, share_uniform=0.5, power=4.0)["name"] for _ in range(4000)]
    assert 0.08 < half.count("dull") / len(half) < 0.4          # ~ 0.5 x 0.5 + 0.5 x 1/17
