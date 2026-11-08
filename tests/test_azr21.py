"""azr21 (Thomas 2026-11-06): lottery vote, the bystanders holding a set share of the vote weight; five doses 0-0.4. Torch-free."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _load  # noqa: E402

VR = _load("vote_rules")
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
P, B = 3, 2
Y, N, RS = VR.PLAY, VR.DECLINE, VR.RESET
DOSES = (0, 10, 20, 30, 40)


def _cfg(name):
    return yaml.safe_load(open(CFG / f"{name}.yaml"))


@pytest.mark.parametrize("share", [0.1, 0.2, 0.3, 0.4, 0.5, 0.57, 0.9])
def test_weight_gives_the_bystanders_their_share(share):
    w = VR.bystander_vote_weight(share, P, B)
    assert w > 0
    assert abs(B * w / (P + B * w) - share) < 1e-12
    p, _ = VR.play_share([N, N, N, Y, Y], P, w)                   # only the bystanders say play: the play share IS their share
    assert abs(p - share) < 1e-12


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_weight_rejects_shares_outside_0_1(bad):
    with pytest.raises(ValueError):
        VR.bystander_vote_weight(bad, P, B)


def test_play_share_bounds_and_analytic_cases():
    w = VR.bystander_vote_weight(0.3, P, B)
    assert VR.play_share([Y, Y, Y, Y, Y], P, w) == (1.0, 0.0)
    assert VR.play_share([N, N, N, N, N], P, w) == (0.0, 0.0)
    p, _ = VR.play_share([Y, Y, Y, N, N], P, w)
    assert abs(p - 0.7) < 1e-12                                     # players unanimous for, bystanders against: 1 - 0.3
    p, r = VR.play_share([RS, RS, RS], P, 0.0)                     # the 0% control: players only
    assert (p, r) == (1.0, 1.0)
    p, _ = VR.play_share([Y, N, N], P, 0.0)
    assert abs(p - 1 / 3) < 1e-12


def test_lottery_draw_and_reset():
    w = VR.bystander_vote_weight(0.3, P, B)
    votes = [Y, Y, Y, N, N]                                          # p_play 0.7
    assert VR.lottery(votes, P, w, 0.69)[0] is True
    p = VR.play_share(votes, P, w)[0]
    assert VR.lottery(votes, P, w, p)[0] is False                   # u < p, so u == p declines
    assert VR.lottery(votes, P, w, 0.71)[0] is False
    go, reset, p = VR.lottery([RS, RS, RS, N, N], P, w, 0.0)
    assert go and reset and abs(p - 0.7) < 1e-12                    # reset weight 0.7 > half
    go, reset, _ = VR.lottery([RS, RS, RS, N, N], P, w, 0.99)
    assert not go and not reset                                     # never reset a game that is not played
    go, reset, _ = VR.lottery([RS, Y, Y, Y, Y], P, w, 0.0)
    assert go and not reset                                         # one reset vote of 1 / (3 + 2w): not a majority


def test_lottery_frequency_matches_the_share():
    import random
    w = VR.bystander_vote_weight(0.2, P, B); rng = random.Random(0); votes = [Y, N, N, Y, Y]
    p = VR.play_share(votes, P, w)[0]
    k = sum(VR.lottery(votes, P, w, rng.random())[0] for _ in range(20000))
    assert abs(k / 20000 - p) < 0.015


def test_majority_unchanged():
    assert VR.majority([Y, Y, Y, N, N]) == (True, False)
    assert VR.majority([Y, Y, N, N, N]) == (False, False)


def test_doses_are_azr20_inf_with_the_lottery():
    base = _cfg("azr20_inf")
    for d in DOSES:
        c = _cfg(f"azr21_s{d:02d}")
        assert c["run_id"] == f"azr21_s{d:02d}" and c["output_base"] == "outputs/infinite_games_azr21"
        v = c["runs"][0]["vote"]
        assert v.pop("lottery_share") == pytest.approx(d / 100)
        assert v.pop("bystanders_vote") is (d > 0)
        assert v["bystander_share"] is True and v["harm_veto"] is False and "bystander_seats" not in v
        assert c["train"].pop("max_minutes_per_run") == 390 and c.pop("max_hours") == 8.0
        b = yaml.safe_load(yaml.safe_dump(base))
        bv = b["runs"][0]["vote"]; assert bv.pop("bystander_seats") == 2 and bv.pop("bystanders_vote") is True
        b["train"].pop("max_minutes_per_run"); b.pop("max_hours")
        for x in (b, c):
            x.pop("run_id"); x.pop("output_base")
        assert b == c, d


def test_azr21b_doses_are_azr21_at_a_fixed_100_updates():
    for d in (10, 20, 30, 40):
        a, b = _cfg(f"azr21_s{d:02d}"), _cfg(f"azr21b_s{d:02d}")
        assert b.pop("run_id") == f"azr21b_s{d:02d}" and a.pop("run_id") == f"azr21_s{d:02d}"
        assert b["train"].pop("steps") == 100 and a["train"].pop("steps") == 6000
        assert b["train"].pop("max_minutes_per_run") is None and a["train"].pop("max_minutes_per_run") == 390
        assert b.pop("max_hours") == 10.5 and a.pop("max_hours") == 8.0
        assert a == b, d


def test_dose_readout_takes_a_late_window(tmp_path):
    import json, subprocess, sys
    run = tmp_path / "r" / "runs" / "main"; run.mkdir(parents=True)
    rows = [{"step": s, "source": "proposal", "p_play1": 1.0, "played1": True, "g1_takers": [],
             "g1": {"bystander_loss": 1.0 if s >= 90 else 0.0}} for s in range(100)]
    (run / "traces.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    script = Path(__file__).resolve().parents[1] / "scripts" / "azr21_dose.py"
    for win, name, want in ((None, "a.json", 0.0), ("90,100", "b.json", 1.0)):
        cmd = [sys.executable, str(script), "--run", f"0.1={tmp_path / 'r'}", "--out", str(tmp_path), "--name", name]
        subprocess.run(cmd + (["--window", win] if win else []), check=True, capture_output=True)
        w = json.load(open(tmp_path / name))["doses"]["0.1"]["window"]
        assert w["drain"][0] == want and w["complete"] is True


def test_smoke_runs_the_control_and_one_dose():
    c = _cfg("azr21_smoke")
    names = {r["name"]: r["vote"] for r in c["runs"]}
    assert set(names) == {"s00", "s30"} and c["plan"] == ["seeds", "tune", "run:s00", "run:s30", "readout"]
    assert names["s00"]["lottery_share"] == 0.0 and names["s00"]["bystanders_vote"] is False
    assert names["s30"]["lottery_share"] == 0.3 and names["s30"]["bystanders_vote"] is True
    assert c["output_base"].startswith("outputs/smoke_runs/")


def _code(path: Path) -> str:
    """Source without comments (test the code, not the prose)."""
    import io
    import tokenize
    toks = [t for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline) if t.type != tokenize.COMMENT]
    return tokenize.untokenize(toks)


def test_wiring_lottery_reaches_the_vote_and_the_trace():
    src = _code(ROOT / "src" / "infinite_games" / "azr_vote.py"); tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "lottery"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[2]) == "w_lot" and "lot_rng.random()" in ast.unparse(calls[0])
    assert 'vc.get("lottery_share")' in src or "vc.get('lottery_share')" in src
    assert '"p_play1": e.get("p_play1")' in src and '"p_play2": e.get("p_play2")' in src


def test_run_scripts_have_no_pipes_and_cover_every_dose():
    for name in ("azr21.sh", "azr21_smoke.sh"):
        body = "\n".join(l for l in (ROOT / "scripts" / name).read_text().splitlines() if not l.lstrip().startswith("#"))
        assert "| tail" not in body and "| head" not in body and "| grep" not in body, name
    run = (ROOT / "scripts" / "azr21.sh").read_text()
    assert "for p in 00 40 20 10 30" in run
    for d in DOSES:
        assert (CFG / f"azr21_s{d:02d}.yaml").exists()


def test_dose_readout_on_a_toy_trace(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import azr21_dose as AD
    run = tmp_path / "r" / "runs" / "main"; run.mkdir(parents=True)
    import json
    tr = []
    for step in range(100):
        drain = step >= 50
        tr.append({"step": step, "source": "proposal", "played1": True, "p_play1": 0.7, "g1_takers": [[] , []] if drain else [[0], []],
                   "g1": {"bystander_loss": 0.5}})
        tr.append({"step": step, "source": "anchor", "played1": True, "g1": {"bystander_loss": 0.9}, "g1_takers": [[], []]})
    (run / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in tr))
    s = AD.summarise(AD.rows(tmp_path / "r"))
    w = s["window"]
    assert w["complete"] and w["drain"][0] == 1.0 and w["drain"][3] == 30          # anchors left out; steps 60-89 all drain
    assert abs(w["p_play1"][0] - 0.7) < 1e-12
    assert s["bins"][0]["drain"][0] == 0.0 and s["bins"][50]["drain"][0] == 1.0
    lo, m, hi = w["loss"][1], w["loss"][0], w["loss"][2]
    assert lo <= m <= hi
