"""azr18 (Thomas 2026-10-26): azr17's three runs with the explicit theft rules removed -- no harm veto, no takers' or voter fines, no
controller aimed at the harmful-game share. Torch-free."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _load  # noqa: E402

VC = _load("vote_control")
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
THEFT_KEYS = ("veto_penalty", "veto_voter_penalty")


def _cfg(name):
    return yaml.safe_load(open(CFG / f"{name}.yaml"))


def _strip(cfg):
    """azr17 config minus the fields azr18 is allowed to change."""
    cfg = dict(cfg); cfg.pop("run_id"); cfg.pop("output_base")
    runs = []
    for r in cfg["runs"]:
        v = dict(r["vote"])
        for k in ("harm_veto",) + THEFT_KEYS:
            v.pop(k, None)
        v["control"] = {k: x for k, x in v["control"].items() if k != "harm"}
        runs.append(dict(r, vote=v))
    cfg["runs"] = runs
    return cfg


def test_controllers_without_harm_have_no_harm_multiplier():
    c = VC.Controllers({"kp": 1.0, "ki": 0.1, "kd": 0.5}, harm=False, max_step=0.05)
    for _ in range(50):
        lam = c.update({"decline": 0.2, "harm": 1.0, "allow": 0.9})
    assert "harm" not in lam
    kn = c.knobs()
    assert kn["taker_scale"] == 1.0
    assert kn["group_scale"] == (1 + lam["stay"]) / (1 + lam["exit"])


def test_controllers_default_keeps_harm():
    c = VC.Controllers({"kp": 1.0, "ki": 0.1, "kd": 0.5})
    for _ in range(50):
        c.update({"decline": 0.2, "harm": 1.0, "allow": 0.9})
    assert c.lambdas()["harm"] > 0 and c.knobs()["taker_scale"] > 1.0




