"""azr16 (configs/azr16_gentle.yaml; 2026-10-20): the rate-limited controller, its wiring, the readout K1-K3, configs and
scripts. Torch-free."""
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
R16 = _load("azr16_readout")
GENTLE = {"kp": 1.0, "ki": 0.1, "kd": 0.5}


def test_step_limit_caps_rise_and_fall():
    c = VC.Controllers(dict(GENTLE), ema=1.0, max_step=0.05)
    vals = [c.update({"decline": 0.95})["exit"] for _ in range(10)]               # a spike far above the 0.30 ceiling
    assert all(b - a <= 0.05 + 1e-9 for a, b in zip([0.0] + vals, vals)) and vals[-1] == pytest.approx(0.5)
    down = [c.update({"decline": 0.2})["exit"] for _ in range(5)]                  # back inside the band: falls at most 0.05 per update
    assert all(a - b <= 0.05 + 1e-9 for a, b in zip([vals[-1]] + down, down)) and down[-1] < vals[-1]
    old = VC.Controllers(dict(VC.PID), ema=1.0)                                    # the azr11-15 rule: no limit
    assert old.update({"decline": 0.95})["exit"] > 0.05 * 10


def test_integral_frozen_while_limited():
    p = VC.PIDLagrangian(1.0, 0.1, 0.0, 0.3, +1, max_step=0.05)
    for _ in range(5):
        p.update(1.0)
    assert p.integral == pytest.approx(0.0)                                       # limited every step: no wind-up behind the limit
    q = VC.PIDLagrangian(1.0, 0.1, 0.0, 0.3, +1)
    q.update(1.0); assert q.integral == pytest.approx(0.07)


def test_defaults_unchanged():
    assert VC.Controllers(dict(VC.PID)).knobs() == VC.fixed_knobs() and VC.PIDLagrangian(1, 0.1, 2, 0.3, 1).max_step is None


def test_wiring():
    av = (PKG / "azr_vote.py").read_text()
    assert 'ema=float(cc.get("ema", VC.EMA))' in av and 'max_step=cc.get("max_step")' in av
    ck = (ROOT / "scripts" / "check_azr3.py").read_text(); assert "max_step set but no controller telemetry" in ck
    for p in [PKG / "azr_vote.py", PKG / "vote_control.py", PKG / "azr16_readout.py", ROOT / "scripts" / "check_azr3.py"]:
        compile(p.read_text(), str(p), "exec")


def _run(d: Path, T=150, swing=False):
    d.mkdir(parents=True)
    tel = []
    for i in range(T):
        dec = (0.85 if 90 <= i < 115 else (0.0 if 115 <= i < 140 else 0.2)) if swing else 0.2 + 0.02 * (i % 3)
        tel.append({"step": i, "decline_share_v1": dec, "played1": 0.9, "valid_share": 0.9, "valid_share_r0": 0.8, "rollbacks": 0,
                    "ctl_exit": (0.3 * (i % 2) if swing else 0.01 * (i % 3)), "ctl_stay": 0.0, "pred_acc_total": 0.4, "fun_share": 0.45})
    (d / "train_telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tel))
    preds, traces, props = [], [], []
    for s in range(T):
        for j in range(4):
            ex = 0.1 * (j - 1.5)
            preds.append({"step": s, "j": j, "seat": 0, "role": "A", "agency": 0.5, "exp_total": ex, "realised": [{"total": ex}]})
            traces.append({"step": s, "j": j, "v1": [("play" if (ex > 0 or swing) else "decline"), "play", "play"]})
            props.append({"step": s, "i": j, "reason": None, "code": f"g{s}{j}", "agency": 0.5, "vote_bonus": 0.1, "round": 0})
    (d / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    (d / "traces.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    (d / "proposals.jsonl").write_text("".join(json.dumps(p) + "\n" for p in props))
    (d / "proposer_attempts.jsonl").write_text(""); (d / "season_eval.json").write_text(json.dumps({"games": []}))


def test_readout(tmp_path):
    rep = {"config": {"readout_roles": {"main": "main"}, "reference_dir": str(tmp_path / "nope"),
                      "runs": [{"vote": {"control": {"max_step": 0.05}}}]}, "runs": {"main": {"spec": {"name": "main"}}}}
    _run(tmp_path / "a" / "runs" / "main")
    P = R16.readout(tmp_path / "a", rep)["predictions"]
    assert {"K1", "K2", "K3", "J1", "H1"} <= set(P) and P["K1"]["held"] is True and P["K2"]["held"] is True and P["K3"]["held"] is True
    _run(tmp_path / "b" / "runs" / "main", swing=True)
    Pb = R16.readout(tmp_path / "b", rep)["predictions"]
    assert Pb["K1"]["held"] is False and Pb["K2"]["held"] is False and Pb["K3"]["held"] is not True




