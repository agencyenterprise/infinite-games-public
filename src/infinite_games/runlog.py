"""Run bookkeeping, copied from casper (src/michelangelo/runlog.py, src/sleeper_agent/run_pipeline.py,
src/psi_measurement/config.py and io_utils.py) so this repo needs nothing else from casper.

    run_main(kind, main)    entry-point wrapper: appends a start record to outputs/run_index.jsonl before the run and a
                            finish record (ok, or failed with the traceback) after it. A start with no finish = killed or
                            still running.
    Telemetry               per-stage status and wall-clock, rewritten live to <run dir>/telemetry.json.
    build_provenance        git commit / branch / dirty flag, package versions, argv, timestamp.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_NAME = "run_index.jsonl"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _git(*args: str) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def write_json(obj: Any, path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)


def _versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "peft", "accelerate", "numpy"):
        try:
            out[pkg] = __import__(pkg).__version__
        except Exception:
            out[pkg] = None
    return out


def build_provenance(config_source: str, extra: Optional[Dict] = None) -> Dict:
    diff = _git("diff", "--stat")
    prov = {"timestamp": _now(), "config_source": str(config_source), "git_commit": _git("rev-parse", "HEAD") or "unknown",
            "git_dirty": bool(diff) if diff is not None else None, "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
            "package_versions": _versions(), "argv": list(sys.argv)}
    if extra:
        prov.update(extra)
    return prov


class Telemetry:
    """Lightweight, live-written per-stage telemetry."""

    def __init__(self, path: Path, header: Dict):
        self.path = path; self.data = dict(header); self.data["stages"] = []; self._flush()

    def _flush(self) -> None:
        write_json(self.data, self.path)

    def record(self, model: str, stage: str, status: str, seconds: float, **extra) -> None:
        entry = {"model": model, "stage": stage, "status": status, "seconds": round(seconds, 2)}
        entry.update(extra); self.data["stages"].append(entry); self._flush()

    def finish(self) -> None:
        self.data["finished"] = _now(); self._flush()


def _append(index: Path, record_: dict) -> None:
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a") as fh:
        fh.write(json.dumps(record_) + "\n"); fh.flush(); os.fsync(fh.fileno())   # survive a kill -9 between write and exit


@contextmanager
def record(outputs_root: Path, kind: str, target: Optional[str] = None, out_dir: Optional[Path] = None):
    """Bracket a run with a start and a finish entry; on an exception the finish entry carries the error, then it re-raises."""
    index = Path(outputs_root) / INDEX_NAME; started = dt.datetime.now(dt.timezone.utc)
    run_id = f"{started.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    _append(index, {"run_id": run_id, "event": "start", "at": _now(), "kind": kind, "target": target,
                    "out_dir": str(out_dir) if out_dir else None, "argv": sys.argv, "host": socket.gethostname(),
                    "pid": os.getpid(), "commit": _git("rev-parse", "HEAD")})
    try:
        yield run_id
    except BaseException as exc:
        _append(index, {"run_id": run_id, "event": "finish", "at": _now(), "kind": kind, "target": target, "status": "failed",
                        "seconds": (dt.datetime.now(dt.timezone.utc) - started).total_seconds(), "error": repr(exc),
                        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]})
        raise
    else:
        _append(index, {"run_id": run_id, "event": "finish", "at": _now(), "kind": kind, "target": target, "status": "ok",
                        "seconds": (dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
                        "out_dir": str(out_dir) if out_dir else None})


def _target_from_argv() -> Optional[str]:
    for flag in ("--config", "--model-id", "--run"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 < len(sys.argv):
                return sys.argv[i + 1]
    return None


def run_main(kind: str, main):
    """Index the attempt, then run; returns main's own value so sys.exit(run_main(...)) keeps the exit code."""
    with record(REPO_ROOT / "outputs", kind, target=_target_from_argv()):
        return main()
