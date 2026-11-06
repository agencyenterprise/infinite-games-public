"""Repo-level checks: every module compiles, nothing still imports casper, every shipped config loads and points at files that
exist, and no run script swallows output. Torch-free."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "infinite_games"


def _code_lines(path: Path):
    return [l for l in path.read_text().splitlines() if not l.lstrip().startswith("#")]


def test_every_module_compiles():
    files = sorted(PKG.glob("*.py")) + sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    assert len(files) > 40
    for f in files:
        compile(f.read_text(), str(f), "exec")


def test_nothing_imports_casper():
    for f in sorted(PKG.glob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
        for n in ast.walk(ast.parse(f.read_text())):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
            assert not any(m == "src" or m.startswith("src.") for m in mods), f"{f.name} imports {mods}"


def test_configs_load_and_their_files_exist():
    cfgs = sorted((ROOT / "configs").glob("azr21_*.yaml"))
    assert {c.stem for c in cfgs} == {"azr21_s00", "azr21_s10", "azr21_s20", "azr21_s30", "azr21_s40", "azr21_smoke"}
    for c in cfgs:
        d = yaml.safe_load(open(c))
        assert (ROOT / d["warmup"]["examples"]).is_file(), c.name
        assert d["output_base"].startswith("outputs/"), c.name
        for s in d["seed_dirs"]:
            assert (PKG / "azr_seeds" / s).is_dir(), (c.name, s)
        assert (PKG / "azr_seeds" / d["anchor_dir"]).is_dir(), c.name


def test_reference_traces_are_shipped():
    for k in ("azr19_inf", "azr20_inf"):
        assert (ROOT / "data" / "reference" / k / "runs" / "main" / "traces.jsonl").stat().st_size > 100_000


def test_sync_only_collects():
    """The sync contract: host -> local, never removes anything local, never pulls weights."""
    f = ROOT / "scripts" / "sync_outputs.sh"; body = "\n".join(_code_lines(f))
    for bad in ("--delete", "--remove-source-files", "--delete-excluded"):
        assert bad not in body, bad
    assert not re.search(r"(^|[;&|]\s*|\s)rm(\s|$)", body, re.M), "rm in an executable line"
    assert re.search(r'rsync [^\n]*"\$host:\$REMOTE_ROOT/" "\$LOCAL_ROOT/"', body), "rsync must run host -> local"
    for pat in ("*.safetensors", "*.bin", "*.pt", "checkpoints/"):
        assert f"--exclude='{pat}'" in body, pat
    assert "/workspace/infinite-games/outputs" in body and "PIPESTATUS[0]" in body


def test_run_scripts_never_swallow_output():
    for f in sorted((ROOT / "scripts").glob("*.sh")):
        body = "\n".join(_code_lines(f))
        assert not re.search(r"\|\s*(tail|head|grep|less)\b", body), f.name
        assert "src.infinite_games" not in body and "/workspace/casper" not in f.read_text(), f.name
