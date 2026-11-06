"""Finite vs infinite games: train an LLM player for winning with and without regard for keeping the other players in
the game, and test it on held-out games, probes and prose dilemmas; psi on every trained policy.

Units run in the order of the config's ``plan`` (``max_hours`` stops new units once spent, so a short night keeps the
most important units -- the first seed's four arms -- whole):

    calibrate   base policy: the full held-out suite (the reference), calibration rollouts on the training family ->
                lam_d for the matched control, and a fingerprint (base probabilities on a few probes) used to check that
                removing an adapter restores the base model exactly
    arms:<s>    for each arm, seed s: fresh LoRA -> GRPO (src/infinite_games/grpo.py) -> save adapter -> held-out suite ->
                unload -> fingerprint check. The first arm to finish fixes the step count for every later arm (if it was
                cut by max_minutes, the others train for the same number of steps, not the same minutes)
    psi:<s>     frees the GPU, then scripts/run_ablation_psi.py on base + every seed-s adapter (attached at load, model
                config ``peft_adapter``), own continuations on the held-out game-state corpus + the joint corpus
    readout     paired comparisons + the registered predictions (src/infinite_games/readout.py) -> readout.json

Outputs: <output_base>/<run_id>_<UTC>/{report.json, telemetry.json, readout.json, calibrate.json, base/, s<seed>/<arm>/
{adapter/, train_telemetry.jsonl, eval.json, items.json}, psi/, logs/}.

    python -m infinite_games.pipeline --config configs/infinite_games.yaml
    python -m infinite_games.pipeline --config configs/infinite_games_smoke.yaml --strict
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infinite_games import engine as E  # noqa: E402
from infinite_games import probes as PB  # noqa: E402
from infinite_games import readout as RO  # noqa: E402
from infinite_games import rewards as RW  # noqa: E402
from infinite_games.runlog import Telemetry, build_provenance  # noqa: E402

LOG = logging.getLogger("infinite_games")
UNIT_KINDS = ("calibrate", "arms", "psi", "readout")
SECONDS_PER_HOUR = 3600.0
FINGERPRINT_N = 16
FINGERPRINT_TOL = 1e-3
PROMPT_DIR = REPO_ROOT / "data" / "psi_prompts"
HEADLINE_KEYS = ["heldout_win", "heldout_ko", "heldout_cripple", "heldout_harm", "heldout_others_capacity", "train_win",
                 "train_ko", "nokill_win", "probe_only_path_heldout_ko", "probe_gratuitous_heldout_ko", "probe_early_heldout_ko",
                 "pair_heldout_final_minus_mid", "vignette_ko_cripple", "vignette_help"]


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    tmp.replace(path)


def game_corpus(path: Path, n: int, seed: int) -> int:
    """Held-out decision prompts as a psi prompt corpus ({"text": ...} per line); deterministic in seed."""
    import random
    rng = random.Random(seed); rows = []
    while len(rows) < n:
        spec = E.sample_spec("heldout", rng); ep = E.Episode(spec, rng.randrange(2 ** 31)); steps = rng.randint(0, 4)
        d = ep.pending()
        while d is not None and steps > 0:
            ep.choose(rng.randrange(d.n_options)); d = ep.pending(); steps -= 1
        if d is not None:
            rows.append({"text": d.prompt})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


class Night:
    def __init__(self, cfg: Dict, cfg_path: Path, strict: bool):
        self.cfg, self.cfg_path, self.strict = cfg, cfg_path, strict
        self.out = REPO_ROOT / cfg["output_base"] / f"{cfg['run_id']}_{_stamp()}"; (self.out / "logs").mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.out / "logs" / "pipeline.log"); fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)
        self.t0 = time.time()
        self.tel = Telemetry(self.out / "telemetry.json", {"run_id": cfg["run_id"], "out_dir": str(self.out),
                                                            "started": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                                                            "provenance": build_provenance(str(cfg_path))})
        self.report: Dict = {"run_id": cfg["run_id"], "out_dir": str(self.out), "config": cfg, "units": [], "arms": {},
                             "psi": {}, "locked_steps": None}
        self.model = self.tok = self.pol = None; self.base_model = None
        ev = cfg["eval"]
        self.probes = PB.build_probes(int(ev["probes_per_type"]), seed=int(ev["probe_seed"]))
        self.vignettes = PB.build_vignettes(int(ev["vignette_fills"]), seed=int(ev["probe_seed"]))
        self.lam_d: Optional[float] = None; self.fingerprint: Optional[List[List[float]]] = None
        self.failures: List[str] = []
        _dump(self.report, self.out / "report.json")

    # ---------------------------------------------------------------------------------------------------- helpers
    def hours(self) -> float:
        return (time.time() - self.t0) / SECONDS_PER_HOUR

    def over_budget(self) -> bool:
        mh = self.cfg.get("max_hours"); return mh is not None and self.hours() > float(mh)

    def unit(self, name: str, fn, budget_exempt: bool = False) -> Optional[Dict]:
        if self.over_budget() and not budget_exempt:
            LOG.warning("skipping %s: max_hours %.2f spent", name, float(self.cfg["max_hours"]))
            self.tel.record(self.cfg["model"]["name"], name, "skipped_budget", 0.0); self.report["units"].append({"unit": name, "status": "skipped_budget"})
            _dump(self.report, self.out / "report.json"); return None
        t = time.time(); LOG.info("=== unit %s (%.2f h elapsed)", name, self.hours())
        try:
            res = fn(); status, err = "ok", None
        except Exception as e:                      # record before deciding whether to re-raise
            res, status, err = None, "failed", {"error": repr(e), "traceback": traceback.format_exc()}
            LOG.error("unit %s FAILED: %s\n%s", name, e, err["traceback"]); self.failures.append(name)
        self.tel.record(self.cfg["model"]["name"], name, status, time.time() - t, **({"error": err["error"]} if err else {}))
        self.report["units"].append({"unit": name, "status": status, "seconds": round(time.time() - t, 1), **(err or {})})
        _dump(self.report, self.out / "report.json")
        if err and self.strict:
            raise RuntimeError(f"unit {name} failed (strict): {err['error']}")
        return res

    def ensure_model(self) -> None:
        if self.base_model is not None:
            return
        import torch
        from infinite_games import policy as PO
        mc = self.cfg["model"]; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOG.info("loading %s", mc["model_id"]); self.base_model, self.tok = PO.load_model(mc["model_id"], dev, mc.get("dtype", "bfloat16"), mc.get("revision"))
        self.base_model.config.use_cache = False
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base_model.enable_input_require_grads()
        drop = {k: v for k, v in self.base_model.config.to_dict().items() if "dropout" in k and isinstance(v, (int, float)) and v}
        if drop:
            raise RuntimeError(f"nonzero dropout {drop}: train() mode would change the policy between rollout and update")
        self.pol = PO.LetterPolicy(self.base_model, self.tok, dev, int(self.cfg["train"]["max_len"]), int(self.cfg["train"]["micro_batch"]))

    def free_model(self) -> None:
        import torch
        self.pol = None; self.base_model = None; self.tok = None; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fingerprint_now(self) -> List[List[float]]:
        ps = self.probes[:FINGERPRINT_N]; return self.pol.probs([p.prompt for p in ps], [p.n_options for p in ps])

    def evaluate(self, where: Path, seed: int) -> Dict:
        from infinite_games import evaluate as EV
        res = EV.evaluate(self.pol, self.cfg["eval"], seed, self.probes, self.vignettes)
        _dump(res["summary"], where / "eval.json"); _dump(res["items"], where / "items.json")
        return res["summary"]

    # -------------------------------------------------------------------------------------------------------- units
    def u_env(self) -> Dict:
        import torch
        pc = self.cfg["psi"]; corpus = PROMPT_DIR / f"{pc['game_corpus']}.jsonl"
        n = game_corpus(corpus, int(pc["game_corpus_n"]), int(pc.get("corpus_seed", 0)))
        lat = REPO_ROOT / "configs" / "lattice_psi" / f"{pc['lattice_base']}_lattice.yaml"
        missing = [str(p) for p in (lat, PROMPT_DIR / f"{pc['joint_corpus']}.jsonl") if not p.exists()]
        if missing and pc.get("enabled", True):
            raise FileNotFoundError(f"psi inputs missing: {missing}")
        info = {"cuda": torch.cuda.is_available(), "game_corpus": str(corpus), "game_corpus_rows": n, "n_probes": len(self.probes),
                "n_vignettes": len(self.vignettes)}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(); info.update(gpu_free_gib=free / 2 ** 30, gpu_total_gib=total / 2 ** 30)
        self.report["env"] = info; return info

    def u_calibrate(self) -> Dict:
        from infinite_games import grpo as GR
        import torch
        self.ensure_model(); base_dir = self.out / "base"
        self.fingerprint = self.fingerprint_now()
        summ = self.evaluate(base_dir, int(self.cfg["eval"]["seed"]))
        cc = self.cfg["calibrate"]; rng_seed = int(cc.get("seed", 12345))
        import random
        rng = random.Random(rng_seed); specs = [E.sample_spec("train", rng) for _ in range(int(cc["episodes"]))]
        eps, _, masses = GR.rollout(self.pol, specs, [rng.randrange(2 ** 31) for _ in specs], torch.Generator().manual_seed(rng_seed),
                                    desc="calibrate")
        outs = [ep.outcome() for ep in eps]
        lam_b = next((float(a.get("lam", 1.0)) for a in self.cfg["arms"] if a["reward"] == "win_minus_ko"), 1.0)
        cal = RW.matched_lambda(outs, lam_b); self.lam_d = cal["lam_d"]
        cal.update({"episodes": len(outs), "base_train_win": sum(o["win"] for o in outs) / len(outs),
                    "legal_mass": sum(masses) / max(1, len(masses)), "fingerprint": self.fingerprint})
        _dump(cal, self.out / "calibrate.json"); self.report["calibrate"] = {k: v for k, v in cal.items() if k != "fingerprint"}
        self.report["base_eval"] = str(base_dir / "eval.json")
        LOG.info("calibration: base ko %.3f / invest %.3f per episode -> lam_d %.3f%s; base legal mass %.3f",
                 cal["ko_rate"], cal["invest_rate"], cal["lam_d"], " (clipped)" if cal["clipped"] else "", cal["legal_mass"])
        return {"lam_d": self.lam_d, "heldout_ko": summ["episodes_heldout"]["ko"]["mean"]}

    def u_arm(self, arm: Dict, seed: int) -> Dict:
        from peft import LoraConfig, get_peft_model
        from infinite_games import grpo as GR
        self.ensure_model()
        if self.lam_d is None:
            cal = json.load(open(self.out / "calibrate.json")); self.lam_d = cal["lam_d"]; self.fingerprint = cal["fingerprint"]
        tc = self.cfg["train"]; lc = tc["lora"]; where = self.out / f"s{seed}" / arm["name"]
        model = get_peft_model(self.base_model, LoraConfig(r=int(lc["rank"]), lora_alpha=int(lc["alpha"]), lora_dropout=0.0,
                                                          target_modules=list(lc["target_modules"]), task_type="CAUSAL_LM"))
        self.pol.model = model
        try:
            n_steps = int(self.report["locked_steps"] or tc["steps"])
            max_min = None if self.report["locked_steps"] else tc.get("max_minutes_per_arm")
            tr = GR.train_arm(self.pol, arm, tc, self.lam_d, seed, where / "train_telemetry.jsonl", n_steps, max_min)
            if self.report["locked_steps"] is None:
                self.report["locked_steps"] = tr["steps_done"]
                LOG.info("step count locked at %d (first arm %s, stopped by %s)", tr["steps_done"], arm["name"], tr["stopped"])
            model.save_pretrained(str(where / "adapter"))
            summ = self.evaluate(where, int(self.cfg["eval"]["seed"]))
        finally:
            self.base_model = model.unload(); self.pol.model = self.base_model
        fp = self.fingerprint_now(); dev = max(abs(a - b) for x, y in zip(fp, self.fingerprint) for a, b in zip(x, y))
        if dev > FINGERPRINT_TOL:
            raise RuntimeError(f"base model not restored after unloading the adapter (max prob change {dev:.2e})")
        rec = {"seed": seed, "reward": arm["reward"], "adapter": str(where / "adapter"), "train": tr, "fingerprint_dev": dev,
               "heldout_ko": summ["episodes_heldout"]["ko"]["mean"], "heldout_win": summ["episodes_heldout"]["win"]["mean"]}
        self.report["arms"][f"s{seed}/{arm['name']}"] = rec; return rec

    def u_psi(self, seed: int) -> Dict:
        raise RuntimeError("the psi unit needs casper (src.cast_psi); it is not part of this repo")
        pc = self.cfg["psi"]; self.free_model(); root = self.out / "psi"; mc = dict(pc["measure"]); mc["output_root"] = str(root)
        corpora = [pc["game_corpus"], pc["joint_corpus"]]; targets = [("base", None)] if seed == self.cfg["plan_psi_base_seed"] else []
        targets += [(k.split("/")[1], v["adapter"]) for k, v in self.report["arms"].items() if k.startswith(f"s{seed}/")]
        if pc.get("max_targets"):
            targets = targets[:int(pc["max_targets"])]
        out = {}
        for name, adapter in targets:
            if self.over_budget():
                out[name] = {"status": "skipped_budget"}; continue
            cell = f"{pc['lattice_base']}_ig_{name}_s{seed}" if adapter else pc["lattice_base"]
            lat = AC.generate_model_configs(pc["lattice_base"], "adapted" if adapter else "base", Path("."), self.out / "psi_configs",
                                            REPO_ROOT, name=cell if adapter else None, model_path=self.cfg["model"]["model_id"] if adapter else None,
                                            model_extra={"peft_adapter": adapter} if adapter else None)
            r = FS.measure(lat, corpora, mc, self.out / "logs" / f"psi_{cell}.log", int(mc["n_prompts"]))
            out[name] = r
            if r["returncode"] != 0 or any(v is None for v in r["runs"].values()):
                self.failures.append(f"psi:{cell}")
        self.report["psi"][f"s{seed}"] = out; return out

    def u_readout(self) -> Dict:
        pols: Dict[str, Dict] = {}
        base_items = self.out / "base" / "items.json"
        if base_items.exists():
            pols["base"] = RO.metrics(json.load(open(base_items)))
        per_seed: Dict[str, Dict] = {}
        for key, rec in self.report["arms"].items():
            sd, name = key.split("/"); items = Path(rec["adapter"]).parent / "items.json"
            per_seed.setdefault(sd, {"base": pols.get("base")} if "base" in pols else {})[name] = RO.metrics(json.load(open(items)))
        res = {"per_seed": {}, "psi": self.report.get("psi")}
        for sd, m in per_seed.items():
            names = {a["name"]: a["reward"] for a in self.cfg["arms"]}
            mm = {("base" if k == "base" else names[k]): v for k, v in m.items()}      # predictions key arms by reward
            res["per_seed"][sd] = {"headline": RO.headline_table(mm, HEADLINE_KEYS), "predictions": RO.predictions(mm, seed=0),
                                   "vs_win": {k: RO.compare(v, mm["win"]) for k, v in mm.items() if "win" in mm and k != "win"}}
        _dump(res, self.out / "readout.json")
        for sd, r in res["per_seed"].items():
            LOG.info("readout %s:", sd)
            for pol, row in r["headline"].items():
                LOG.info("  %-18s %s", pol, "  ".join(f"{k}={v:.3f}" for k, v in row.items() if v is not None))
            for p, v in r["predictions"].items():
                LOG.info("  %-6s held=%s", p, v["held"])
        return {"seeds": list(res["per_seed"])}

    # --------------------------------------------------------------------------------------------------------- run
    def run(self, kinds: List[str]) -> int:
        self.unit("env", self.u_env)
        for item in self.cfg["plan"]:
            kind, _, arg = str(item).partition(":")
            if kind not in UNIT_KINDS:
                raise ValueError(f"unknown plan item {item!r}")
            if kind not in kinds:
                continue
            if kind == "calibrate":
                self.unit("calibrate", self.u_calibrate)
            elif kind == "arms":
                for arm in self.cfg["arms"]:
                    self.unit(f"arm:{arm['name']}:s{arg}", lambda arm=arm: self.u_arm(arm, int(arg)))
            elif kind == "psi":
                if self.cfg["psi"].get("enabled", True):
                    self.unit(f"psi:s{arg}", lambda: self.u_psi(int(arg)))
            elif kind == "readout":
                self.unit("readout", self.u_readout, budget_exempt=True)
        self.tel.finish(); self.report["failures"] = self.failures; self.report["hours"] = round(self.hours(), 3)
        _dump(self.report, self.out / "report.json")
        LOG.info("done in %.2f h -> %s (failures: %s)", self.hours(), self.out, self.failures or "none")
        return 1 if (self.failures and self.strict) else 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True); ap.add_argument("--kinds", default=",".join(UNIT_KINDS))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]; bad = [k for k in kinds if k not in UNIT_KINDS]
    if bad:
        ap.error(f"unknown unit kind(s): {bad} (valid: {UNIT_KINDS})")
    cfg = yaml.safe_load(open(a.config))
    return Night(cfg, a.config, a.strict).run(kinds)


if __name__ == "__main__":
    from infinite_games.runlog import run_main
    sys.exit(run_main("infinite_games", main) or 0)
