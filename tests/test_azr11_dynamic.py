"""azr11, dynamic reward weights (configs/azr11_dynamic.yaml; 2026-10-12): the PID-Lagrangian controllers, the rules they
drive (decline cost, Pyrrhic strength), the wiring, the readout D1-D7, configs and scripts. Torch-free."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
CFG = ROOT / "configs"
VC = _load("vote_control")
VR = _load("vote_rules")
R11 = _load("azr11_readout")


# ------------------------------------------------------------------------------------------------------------- controllers
def test_pid_lagrangian_upper_bound():
    c = VC.PIDLagrangian(kp=1.0, ki=0.1, kd=2.0, threshold=0.3, sense=+1)
    assert c.update(0.2) == 0.0 and c.integral == 0.0                     # satisfied: nothing, integral clamped at 0 (anti-windup)
    v1 = c.update(0.5); assert v1 == pytest.approx(1.0 * 0.2 + 0.02 + 2.0 * 0.3)   # P + I + D (error rose from -0.1 to 0.2)
    v2 = c.update(0.5); assert v2 == pytest.approx(0.2 + 0.04)            # D acts only on a WORSENING error
    assert c.update(None) == v2                                           # nothing measured: hold
    for _ in range(500):
        c.update(1.0)
    assert c.value == VC.LAMBDA_MAX and c.integral == VC.LAMBDA_MAX       # clipped


def test_pid_lagrangian_lower_bound_and_integral_only():
    c = VC.PIDLagrangian(threshold=0.1, sense=-1, **VC.INTEGRAL_ONLY)
    assert c.update(0.0) == pytest.approx(0.01) and c.update(0.0) == pytest.approx(0.02)      # pure dual ascent
    assert c.update(0.5) == pytest.approx(0.0)                                                 # integral unwinds to 0, not below


def test_controllers_map_statistics_to_knobs():
    k0 = VC.Controllers(dict(VC.PID)).knobs(); assert k0 == VC.fixed_knobs()                  # starts at the azr10 rules
    c = VC.Controllers(dict(VC.PID), ema=1.0)
    lam = c.update({"decline": 0.9, "harm": 0.6, "allow": 0.1})
    assert lam["exit"] > 0 and lam["stay"] == 0 and lam["harm"] > 0 and lam["byst"] > 0
    k = c.knobs()
    assert k["decline_cost"] == lam["exit"] and k["decline_bonus"] == 0
    assert k["taker_scale"] == pytest.approx(1 + lam["harm"]) and k["growth_scale"] == pytest.approx(1 + lam["byst"])
    assert k["group_scale"] == pytest.approx((1 + lam["harm"]) / (1 + lam["exit"]))
    c2 = VC.Controllers(dict(VC.PID), ema=1.0); c2.update({"decline": 0.0, "harm": 0.0, "allow": 0.9})
    assert c2.knobs()["decline_bonus"] > 0 and c2.knobs()["group_scale"] > 1 and c2.knobs()["decline_cost"] == 0


def test_ema_smoothing():
    c = VC.Controllers(dict(VC.INTEGRAL_ONLY), ema=0.5); c.update({"decline": 1.0}); c.update({"decline": 0.0})
    assert c.smoothed["decline"] == pytest.approx(0.5)


# ------------------------------------------------------------------------------------------------------------- rules
def test_decline_cost_and_pyrrhic_strength():
    v = VR.Voter(base=0.05, cost=0.2); assert v.vote(VR.DECLINE) == pytest.approx(-0.15) and v.vote(VR.DECLINE) == pytest.approx(0.025 - 0.2)
    r, g, votes = [0.5, 0.1, 0.2], [0.4, -0.1, 0.0], ["decline", "decline", "play"]
    assert VR.pyrrhic(r, g, votes)[0] == VR.pyrrhic(r, g, votes, strength=1.0)[0]               # default = night-8 rule
    assert VR.pyrrhic(r, g, votes, strength=0.5)[0][0] == pytest.approx(0.5 - 0.4 - 0.2)
    assert VR.pyrrhic(r, g, votes, strength=2.0)[0][1] == pytest.approx(0.1)                    # losers untouched


def test_wiring():
    av = (PKG / "azr_vote.py").read_text()
    # every statistic the controllers read must be computed BEFORE ctl.update (smoke 2026-10-13: decline_share_v1 was set after it)
    i = av.index("ctl.update(")
    for k in ('row["decline_share_v1"] =', 'row["harm_g1_share"] =', "row = telemetry_row("):
        assert av.index(k) < i, k
    for k in ("VC.Controllers(", "VC.fixed_knobs()", "ctl.update(", "strength=kn[\"group_scale\"]", "veto_pen_k", "veto_vpen_k", "growth_k",
              'kn["decline_bonus"]', 'kn["decline_cost"]', 'f"knob_{k}"', 'f"ctl_{n}"'):
        assert k in av, k
    for p in [PKG / "azr_vote.py", PKG / "vote_rules.py", PKG / "vote_control.py", PKG / "azr11_readout.py", ROOT / "scripts" / "check_azr3.py"]:
        compile(p.read_text(), str(p), "exec")


# ------------------------------------------------------------------------------------------------------------- readout
def _run(d: Path, decline, played, harm_first, harm_last, wobble, n=30):
    d.mkdir(parents=True)
    tel = [{"step": i, "decline_share_v1": decline + (wobble if i % 2 else -wobble), "played1": played, "bystander_allow": 0.6,
            "harm_g1_share": harm_first if i < n // 3 else harm_last, "ctl_smoothed_decline": decline, "ctl_smoothed_harm": harm_last,
            "ctl_smoothed_allow": 0.6, "ctl_exit": 0.1, "ctl_stay": 0.0, "ctl_harm": 0.2, "ctl_byst": 0.0, "knob_decline_cost": 0.1,
            "knob_group_scale": 1.1} for i in range(n)]
    (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
    (d / "traces.jsonl").write_text("")
    (d / "season_eval.json").write_text(json.dumps({"games": []}))


def test_readout(tmp_path):
    _run(tmp_path / "runs" / "pid_s0", 0.2, 0.8, 0.7, 0.3, 0.01); _run(tmp_path / "runs" / "pid_s1", 0.25, 0.75, 0.6, 0.25, 0.02)
    _run(tmp_path / "runs" / "integral", 0.3, 0.7, 0.6, 0.4, 0.2)
    report = {"config": {"readout_roles": {n: n for n in ("pid_s0", "pid_s1", "integral")}, "reference_dir": str(tmp_path / "nope")},
              "runs": {n: {"spec": {"name": n}} for n in ("pid_s0", "pid_s1", "integral")}}
    P = R11.readout(tmp_path, report)["predictions"]
    assert P["D1"]["held"] is True and P["D2"]["held"] is True and P["D3"]["held"] is True and P["D4"]["held"] is True
    assert set(P) == {"D1", "D2", "D3", "D4", "D5", "D6", "D7"}


# ------------------------------------------------------------------------------------------------------------- configs, scripts


