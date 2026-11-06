"""Long sanction run (configs/azr7_long.yaml; 2026-10-04): no reflections, tuned batch sizes, cached base scores, adapter
checkpoints, three guilty-pays regimes. Torch-dependent pieces skip on the Mac."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_azr_games import _check_calls, _leaf_keys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"
CONFIGS = [ROOT / "configs" / "azr7_long.yaml", ROOT / "configs" / "azr7_long_smoke.yaml"]






def test_wiring():
    pol = (PKG / "policy.py").read_text(); a3 = (PKG / "azr3.py").read_text(); pl = (PKG / "azr3_pipeline.py").read_text()
    assert "self.infer_batch" in pol and "B = self.infer_batch" in pol
    assert "base_logp(pol, nets, base_cache" in a3 and 'nets.use("base")' in a3 and "hashlib.sha1" in a3 and "save_every" in a3
    assert '"tune"' in pl and "def u_tune" in pl and "BT.tune(" in pl
    # 2026-10-04 smoke failure: per-folder game sets were filled only with several seed dirs; azr7 has ONE and names it per run
    i = pl.index("tr_, pa_ = self.seed_sets.setdefault"); guard = pl[pl.rindex("\n", 0, pl.rindex("\n", 0, i)):i]
    assert 'self.cfg.get("seed_dirs")' in guard and "multi" not in guard
    assert _check_calls(PKG / "azr3_pipeline.py", "BT", [PKG / "batch_tune.py"]) >= 1
    chk = (ROOT / "scripts" / "check_azr3.py").read_text()
    assert "batch_tune" in chk and "checkpoints" in chk
    for p in (PKG / "batch_tune.py", PKG / "azr3.py", PKG / "policy.py", PKG / "azr3_pipeline.py", ROOT / "scripts" / "check_azr3.py"):
        compile(p.read_text(), str(p), "exec")


def test_base_cache_keys():
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT))
    from infinite_games import azr3 as A3
    k = A3._base_key("sys", "prompt", 3)
    assert k == A3._base_key("sys", "prompt", 3) and len(k) == 20
    assert len({k, A3._base_key("sys", "prompt", 4), A3._base_key(None, "prompt", 3), A3._base_key("sys", "prompt2", 3)}) == 4
