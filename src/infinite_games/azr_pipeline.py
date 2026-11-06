"""Absolute Zero for games, the night: one Qwen2.5-Coder-3B proposes games as code and plays them, across runs that
switch the finite / infinite conditions on and off; everything is evaluated on paired held-out panels; psi at the end.

Units, in the config's ``plan`` order (``max_hours`` stops new units; the readout always runs):
    env         CUDA, the psi config's architecture checked against the downloaded model config, seed files present
    seeds       every seed game validated in the sandbox (train_* -> the proposer's starting buffer, heldout_* -> the
                evaluation panel); the held-out game-state psi corpus is written from random play of the held-out seeds
    base_eval   the untrained model: proposer_eval, season_eval on the seed panel, probe_eval
    run:<name>  fresh LoRA -> azr.train_run(arm, joint, seed, conditions) -> adapter + per-run eval (proposer_eval;
                season_eval + probe_eval for runs whose solver trains) -> unload -> fingerprint check. The first run
                to finish fixes the step count for every later run when it was cut by time
    cross_eval  every trained solver and base on a panel of LATE proposals collected from all runs
    psi         base + the runs marked ``psi: true``, on the held-out game-state corpus and the joint corpus
    readout     azr_readout.py -> readout.json (training summaries, paired comparisons, registered predictions)

    python -m infinite_games.azr_pipeline --config configs/azr_games.yaml
    python -m infinite_games.azr_pipeline --config configs/azr_games_smoke.yaml --strict
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import glob
import json
import logging
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infinite_games import azr_economy as EC  # noqa: E402
from infinite_games import azr_sandbox as SB  # noqa: E402
from infinite_games import probes as PB  # noqa: E402
from infinite_games.runlog import Telemetry, build_provenance  # noqa: E402

LOG = logging.getLogger("azr_games")
SEED_DIR = REPO_ROOT / "src" / "infinite_games" / "azr_seeds"
PROMPT_DIR = REPO_ROOT / "data" / "psi_prompts"
UNIT_KINDS = ("seeds", "base_eval", "warmup", "run", "cross_eval", "psi", "readout")
COMPACT_DIR = SEED_DIR / "compact"
SECONDS_PER_HOUR = 3600.0
FINGERPRINT_N = 8
FINGERPRINT_TOL = 1e-3
ARCH_FIELDS = {"n_layers": "num_hidden_layers", "d_model": "hidden_size", "n_heads": "num_attention_heads"}
LATE_FRACTION = 0.25


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    tmp.replace(path)


def late_proposals(run_dir: Path, k: int) -> List[Dict]:
    """Up to ``k`` valid proposals from the last LATE_FRACTION of a run's steps, highest proposer reward first (any
    step's if the late part has none)."""
    rows = [json.loads(l) for l in open(run_dir / "proposals.jsonl")] if (run_dir / "proposals.jsonl").exists() else []
    if not rows:
        return []
    last = max(r["step"] for r in rows); cut = last - max(1, int(LATE_FRACTION * (last + 1))) + 1
    valid = [r for r in rows if r["reason"] is None and r["code"]]
    ok = sorted([r for r in valid if r["step"] >= cut] or valid, key=lambda r: -r["reward"])      # all valid if none late
    seen, out = set(), []
    for r in ok:
        if r["code"] in seen:
            continue
        seen.add(r["code"]); out.append({"code": r["code"], "source": f"{run_dir.name}_step{r['step']}"})
        if len(out) >= k:
            break
    return out


class Night:
    def __init__(self, cfg: Dict, cfg_path: Path, strict: bool):
        self.cfg, self.strict = cfg, strict
        self.out = REPO_ROOT / cfg["output_base"] / f"{cfg['run_id']}_{_stamp()}"; (self.out / "logs").mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.out / "logs" / "pipeline.log"); fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)
        self.t0 = time.time()
        self.tel = Telemetry(self.out / "telemetry.json", {"run_id": cfg["run_id"], "out_dir": str(self.out),
                                                            "started": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                                                            "provenance": build_provenance(str(cfg_path))})
        self.report: Dict = {"run_id": cfg["run_id"], "out_dir": str(self.out), "config": cfg, "units": [], "runs": {},
                             "locked_steps": None, "psi": {}, "failures": []}
        ev = cfg["eval"]
        self.probes = PB.build_probes(int(ev["probes_per_type"]), seed=int(ev["probe_seed"]))
        self.vignettes = PB.build_vignettes(int(ev["vignette_fills"]), seed=int(ev["probe_seed"]))
        self.model = self.tok = self.pol = None; self.pool = self.server = None
        self.train_seeds: List[Dict] = []; self.panel: List[Dict] = []; self.ref_seeds: List[Dict] = []; self.fingerprint = None
        self.warm_adapter: Optional[str] = None
        _dump(self.report, self.out / "report.json")

    # ---------------------------------------------------------------------------------------------------- helpers
    def hours(self) -> float:
        return (time.time() - self.t0) / SECONDS_PER_HOUR

    def over_budget(self) -> bool:
        mh = self.cfg.get("max_hours"); return mh is not None and self.hours() > float(mh)

    def unit(self, name: str, fn, budget_exempt: bool = False):
        if self.over_budget() and not budget_exempt:
            LOG.warning("skipping %s: max_hours spent", name); self.tel.record(self.cfg["model"]["name"], name, "skipped_budget", 0.0)
            self.report["units"].append({"unit": name, "status": "skipped_budget"}); _dump(self.report, self.out / "report.json"); return None
        t = time.time(); LOG.info("=== unit %s (%.2f h elapsed)", name, self.hours())
        try:
            res = fn(); status, err = "ok", None
        except Exception as e:
            res, status, err = None, "failed", {"error": repr(e), "traceback": traceback.format_exc()}
            LOG.error("unit %s FAILED: %s\n%s", name, e, err["traceback"]); self.report["failures"].append(name)
        self.tel.record(self.cfg["model"]["name"], name, status, time.time() - t, **({"error": err["error"]} if err else {}))
        self.report["units"].append({"unit": name, "status": status, "seconds": round(time.time() - t, 1), **(err or {})})
        _dump(self.report, self.out / "report.json")
        if err and self.strict:
            raise RuntimeError(f"unit {name} failed (strict): {err['error']}")
        return res

    def sandbox(self):
        if self.pool is None:
            sc = self.cfg["sandbox"]; self.pool = SB.Pool(int(sc["validate_workers"])); self.server = SB.GameServer(int(sc["game_workers"]))
        return self.pool, self.server

    def ensure_model(self):
        if self.model is not None:
            return
        import torch
        from infinite_games import policy as PO
        mc = self.cfg["model"]; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOG.info("loading %s", mc["model_id"]); self.model, self.tok = PO.load_model(mc["model_id"], dev, mc.get("dtype", "bfloat16"))
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); self.model.enable_input_require_grads()
        drop = {k: v for k, v in self.model.config.to_dict().items() if "dropout" in k and isinstance(v, (int, float)) and v}
        if drop:
            raise RuntimeError(f"nonzero dropout {drop}")
        tc = self.cfg["train"]; self.pol = PO.LetterPolicy(self.model, self.tok, dev, int(tc["max_len"]), int(tc["micro_batch"]))

    def free_model(self):
        import torch
        self.model = self.tok = self.pol = None; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fingerprint_now(self):
        ps = self.probes[:FINGERPRINT_N]; return self.pol.probs([p.prompt for p in ps], [p.n_options for p in ps])

    def check_restored(self):
        fp = self.fingerprint_now(); dev = max(abs(a - b) for x, y in zip(fp, self.fingerprint) for a, b in zip(x, y))
        if dev > FINGERPRINT_TOL:
            raise RuntimeError(f"base model not restored after unloading the adapter (max prob change {dev:.2e})")
        return dev

    def evaluate(self, where: Path, ref: bool, solver: bool, proposer: bool, cross: bool = False) -> Dict:
        from infinite_games import azr_eval as AE
        pool, server = self.sandbox(); ev = dict(self.cfg["eval"]); seed = int(ev["seed"]); out = {}
        ev["think"] = self.cfg["train"].get("think") if self.cfg["train"].get("think", {}).get("enabled", True) else None
        if cross:                                           # the cross panel is larger, so its seasons are shorter
            ev.update(seasons=ev["cross_seasons"], season_games=ev["cross_season_games"])
        if proposer:
            r = AE.proposer_eval(self.model, self.tok, self.pol, self.train_seeds, self.cfg["train"] | {"learn_instances": ev["learn_instances"], "think": ev["think"]},
                                 int(ev["proposer_games"]), seed, pool, server)
            _dump(r, where / "proposer_eval.json"); out["proposer_valid"] = sum(i["ok"] for i in r["items"]) / max(1, len(r["items"]))
        if solver:
            r = AE.season_eval(self.pol, ref, self.panel, ev, seed, server)
            _dump({k: v for k, v in r.items() if k != "traces"}, where / "season_eval.json")
            with open(where / "season_traces.jsonl", "w") as f:
                for t in r["traces"]:
                    f.write(json.dumps(t) + "\n")
            ok = [g for g in r["games"] if "error" not in g]
            out["collapse_rate"] = sum(g["collapse"] for g in ok) / max(1, len(ok))
            for kind in ("novice", "restrained"):
                cs = [c["success"] for c in r["competence"] if c.get("kind", "novice") == kind]
                out["competence" if kind == "novice" else "competence_restrained"] = sum(cs) / max(1, len(cs))
            _dump(AE.probe_eval(self.pol, ref, self.probes, self.vignettes), where / "probe_eval.json")
        return out

    # -------------------------------------------------------------------------------------------------------- units
    def u_env(self) -> Dict:
        import torch
        from transformers import AutoConfig
        mc = self.cfg["model"]; info = {"cuda": torch.cuda.is_available()}
        seeds = sorted(glob.glob(str(SEED_DIR / "*.py")))
        info["train_seed_files"] = sum("/train_" in s for s in seeds); info["heldout_seed_files"] = sum("/heldout_" in s for s in seeds)
        if not info["train_seed_files"] or not info["heldout_seed_files"]:
            raise FileNotFoundError(f"seed games missing under {SEED_DIR}")
        pc = self.cfg["psi"]
        if pc.get("enabled", True):
            lat = REPO_ROOT / "configs" / "lattice_psi" / f"{pc['lattice_base']}_lattice.yaml"
            psi_cfg = REPO_ROOT / "configs" / "psi_measurement" / f"{pc['lattice_base']}_consciousness_prompts.yaml"
            for p in (lat, psi_cfg, PROMPT_DIR / f"{pc['joint_corpus']}.jsonl"):
                if not p.exists():
                    raise FileNotFoundError(str(p))
            want = yaml.safe_load(open(psi_cfg))["model"]; have = AutoConfig.from_pretrained(mc["model_id"])
            bad = {k: (want[k], getattr(have, v, None)) for k, v in ARCH_FIELDS.items() if k in want and want[k] != getattr(have, v, None)}
            if bad:
                raise ValueError(f"psi config architecture differs from the model's config: {bad}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(); info.update(gpu_free_gib=free / 2 ** 30, gpu_total_gib=total / 2 ** 30)
        self.report["env"] = info; return info

    def u_seeds(self) -> Dict:
        pool, server = self.sandbox(); files = sorted(glob.glob(str(SEED_DIR / "*.py"))); codes = [open(f).read() for f in files]
        verdicts = pool.validate_many(codes, seed=0); bad = [(Path(f).name, v["reason"], v.get("detail", "")[:200]) for f, v in zip(files, verdicts) if not v["ok"]]
        if bad:
            raise RuntimeError(f"seed games fail validation: {bad}")
        for f, c, v in zip(files, codes, verdicts):
            item = {"code": c, "rules": v["stats"]["rules"], "name": Path(f).stem}
            (self.train_seeds if Path(f).name.startswith("train_") else self.panel).append(item)
        self.panel = self.panel + self.train_seeds                  # held-out first, then the training seeds
        cfiles = sorted(glob.glob(str(COMPACT_DIR / "*.py")))
        if cfiles:                                                  # short examples: buffer and warm start only, not the panel
            cv = pool.validate_many([open(f).read() for f in cfiles], seed=0)
            bad = [(Path(f).name, v["reason"]) for f, v in zip(cfiles, cv) if not v["ok"]]
            if bad:
                raise RuntimeError(f"compact seed games fail validation: {bad}")
            self.ref_seeds = [{"code": open(f).read(), "rules": v["stats"]["rules"], "name": Path(f).stem} for f, v in zip(cfiles, cv)]
        n = self.game_corpus(int(self.cfg["psi"]["game_corpus_n"]))
        stats = {Path(f).stem: {k: v["stats"][k] for k in ("turns", "agency", "bystander_temptation", "ruin_temptation", "distinct_moves", "code_lines")}
                 for f, v in zip(files, verdicts)}
        _dump(stats, self.out / "seeds.json")
        return {"train": len(self.train_seeds), "panel": len(self.panel), "compact": len(self.ref_seeds), "psi_corpus_rows": n}

    def game_corpus(self, n: int) -> int:
        """Solver prompts at random states of the held-out seeds (random play), as a psi prompt corpus."""
        from infinite_games import azr as AZ
        _, server = self.sandbox(); rng = random.Random(int(self.cfg["psi"].get("corpus_seed", 0))); held = [p for p in self.panel if p not in self.train_seeds]
        rows = []; tries = 0
        while len(rows) < n and tries < 20 * n:
            tries += 1; g = held[rng.randrange(len(held))]; gid = 10 ** 6 + tries
            ob = server.step({gid: {"op": "new", "code": g["code"], "n": SB.N_PLAYERS, "endowments": [SB.DEFAULT_ENDOWMENT] * SB.N_PLAYERS,
                                    "bystanders": [SB.DEFAULT_BYSTANDER] * SB.N_BYSTANDERS, "seed": rng.randrange(2 ** 31)}})[gid]
            for _ in range(rng.randint(0, 12)):
                if "error" in ob or ob["over"]:
                    break
                ob = server.step({gid: {"op": "play", "player": ob["player"], "move": rng.choice(ob["moves"])}})[gid]
            if "error" not in ob and not ob["over"]:
                rows.append({"text": AZ.solver_prompt(g["rules"], ob["describe"], ob["moves"], ob["player"], EC.world_text(EC.World()))})
            server.step({gid: {"op": "drop"}})
        path = PROMPT_DIR / f"{self.cfg['psi']['game_corpus']}.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return len(rows)

    def u_base_eval(self) -> Dict:
        self.ensure_model(); self.fingerprint = self.fingerprint_now()
        return self.evaluate(self.out / "base", ref=False, solver=True, proposer=True)

    def u_warmup(self) -> Dict:
        """One supervised pass teaching the proposer the interface on seed games (shared starting adapter for every run)."""
        from peft import LoraConfig, get_peft_model
        from infinite_games import azr as AZ
        self.ensure_model(); tc = self.cfg["train"]; wc = self.cfg["warmup"]; lc = tc["lora"]; where = self.out / "warm"
        if self.fingerprint is None:
            self.fingerprint = self.fingerprint_now()
        model = get_peft_model(self.model, LoraConfig(r=int(lc["rank"]), lora_alpha=int(lc["alpha"]), lora_dropout=0.0,
                                                     target_modules=list(lc["target_modules"]), task_type="CAUSAL_LM"))
        self.model = model; self.pol.model = model
        try:
            pairs = AZ.sft_pairs(self.tok, self.train_seeds + self.ref_seeds, int(wc["examples"]), int(tc["n_refs"]),
                                 random.Random(int(wc.get("seed", 0))), int(tc["max_ref_chars"]))
            sft = AZ.sft_warmup(model, self.tok, pairs, wc, where / "sft_log.jsonl")
            model.save_pretrained(str(where / "adapter")); self.warm_adapter = str(where / "adapter")
            summ = self.evaluate(where, ref=False, solver=True, proposer=True)
        finally:
            self.model = model.unload(); self.pol.model = self.model
        dev = self.check_restored(); self.report["warm"] = {"adapter": self.warm_adapter, "sft": sft, "eval": summ, "fingerprint_dev": dev}
        return self.report["warm"]

    def u_run(self, spec: Dict) -> Dict:
        from peft import LoraConfig, get_peft_model
        from infinite_games import azr as AZ
        self.ensure_model(); pool, server = self.sandbox()
        if self.fingerprint is None:
            self.fingerprint = self.fingerprint_now()
        tc = self.cfg["train"]; lc = tc["lora"]; where = self.out / "runs" / spec["name"]
        if self.warm_adapter:                                       # every run starts from the same warm-started adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(self.model, self.warm_adapter, is_trainable=True)
        else:
            model = get_peft_model(self.model, LoraConfig(r=int(lc["rank"]), lora_alpha=int(lc["alpha"]), lora_dropout=0.0,
                                                         target_modules=list(lc["target_modules"]), task_type="CAUSAL_LM"))
        base = self.model; self.model = model; self.pol.model = model
        inject = (SEED_DIR / self.cfg["smoke_inject"]).read_text() if self.cfg.get("smoke_inject") else None
        try:
            n_steps = int(self.report["locked_steps"] or tc["steps"]); max_min = None if self.report["locked_steps"] else tc.get("max_minutes_per_run")
            tr = AZ.train_run(model, self.tok, self.pol, self.train_seeds + self.ref_seeds, tc, spec["arm"], bool(spec.get("joint", True)), int(spec["seed"]),
                              where, n_steps, max_min, pool, server, conditions=spec.get("conditions"), inject=inject)
            if self.report["locked_steps"] is None:
                self.report["locked_steps"] = tr["steps_done"]; LOG.info("step count locked at %d (%s)", tr["steps_done"], tr["stopped"])
            model.save_pretrained(str(where / "adapter"))
            summ = self.evaluate(where, ref=not spec.get("joint", True), solver=bool(spec.get("joint", True)), proposer=True)
        finally:
            self.model = model.unload(); self.pol.model = self.model
        dev = self.check_restored()
        rec = {"spec": spec, "adapter": str(where / "adapter"), "train": tr, "eval": summ, "fingerprint_dev": dev}
        self.report["runs"][spec["name"]] = rec; return rec

    def u_cross_eval(self) -> Dict:
        from peft import PeftModel
        self.ensure_model(); k = int(self.cfg["eval"]["late_per_run"]); panel = []
        for name in self.report["runs"]:
            panel += late_proposals(self.out / "runs" / name, k)
        pool, _ = self.sandbox(); verdicts = pool.validate_many([p["code"] for p in panel], seed=1)
        panel = [dict(p, rules=v["stats"]["rules"]) for p, v in zip(panel, verdicts) if v["ok"]]
        _dump([{k2: p[k2] for k2 in ("source",)} for p in panel], self.out / "cross_panel.json")
        if not panel:
            raise RuntimeError("no late proposals to cross-evaluate")
        own_panel, self.panel = self.panel, panel; out = {}
        try:
            out["base"] = self.evaluate(self.out / "cross" / "base", ref=False, solver=True, proposer=False, cross=True)
            for name, rec in self.report["runs"].items():
                if not rec["spec"].get("joint", True) or self.over_budget():
                    continue
                m = PeftModel.from_pretrained(self.model, rec["adapter"]); base = self.model; self.model = m; self.pol.model = m
                try:
                    out[name] = self.evaluate(self.out / "cross" / name, ref=False, solver=True, proposer=False, cross=True)
                finally:
                    self.model = m.unload(); self.pol.model = self.model
        finally:
            self.panel = own_panel
        self.report["cross"] = out; return out

    def u_psi(self) -> Dict:
        raise RuntimeError("the psi unit needs casper (src.cast_psi); it is not part of this repo")
        pc = self.cfg["psi"]; self.free_model(); mc = dict(pc["measure"]); mc["output_root"] = str(self.out / "psi")
        corpora = [pc["game_corpus"], pc["joint_corpus"]]
        targets = [("base", None)] + ([("warm", self.warm_adapter)] if self.warm_adapter else []) + \
            [(n, r["adapter"]) for n, r in self.report["runs"].items() if r["spec"].get("psi")]
        out = {}
        for name, adapter in targets:
            if self.over_budget():
                out[name] = {"status": "skipped_budget"}; continue
            cell = f"{pc['lattice_base']}_azr_{name}" if adapter else pc["lattice_base"]
            lat = AC.generate_model_configs(pc["lattice_base"], "adapted" if adapter else "base", Path("."), self.out / "psi_configs", REPO_ROOT,
                                            name=cell if adapter else None, model_path=self.cfg["model"]["model_id"] if adapter else None,
                                            model_extra={"peft_adapter": adapter} if adapter else None)
            r = FS.measure(lat, corpora, mc, self.out / "logs" / f"psi_{cell}.log", int(mc["n_prompts"])); out[name] = r
            if r["returncode"] != 0 or any(v is None for v in r["runs"].values()):
                self.report["failures"].append(f"psi:{cell}")
        self.report["psi"] = out; return {k: v.get("returncode") for k, v in out.items()}

    def u_readout(self) -> Dict:
        from infinite_games import azr_readout as AR
        res = AR.readout(self.out, self.report); _dump(res, self.out / "readout.json")
        for k, v in res.get("predictions", {}).items():
            LOG.info("  %-4s held=%s  %s", k, v.get("held"), v.get("claim", ""))
        return {"predictions": {k: v.get("held") for k, v in res.get("predictions", {}).items()}}

    def run(self, kinds: List[str]) -> int:
        self.unit("env", self.u_env)
        try:
            for item in self.cfg["plan"]:
                kind, _, arg = str(item).partition(":")
                if kind not in UNIT_KINDS:
                    raise ValueError(f"unknown plan item {item!r}")
                if kind not in kinds:
                    continue
                if kind == "seeds":
                    self.unit("seeds", self.u_seeds)
                elif kind == "base_eval":
                    self.unit("base_eval", self.u_base_eval)
                elif kind == "warmup":
                    self.unit("warmup", self.u_warmup)
                elif kind == "run":
                    spec = next(r for r in self.cfg["runs"] if r["name"] == arg)
                    self.unit(f"run:{arg}", lambda spec=spec: self.u_run(spec))
                elif kind == "cross_eval":
                    self.unit("cross_eval", self.u_cross_eval)
                elif kind == "psi":
                    if self.cfg["psi"].get("enabled", True):
                        self.unit("psi", self.u_psi)
                elif kind == "readout":
                    self.unit("readout", self.u_readout, budget_exempt=True)
        finally:
            if self.pool is not None:
                self.pool.close(); self.server.close()
        self.tel.finish(); self.report["hours"] = round(self.hours(), 3); _dump(self.report, self.out / "report.json")
        LOG.info("done in %.2f h -> %s (failures: %s)", self.hours(), self.out, self.report["failures"] or "none")
        return 1 if (self.report["failures"] and self.strict) else 0


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
    names = [r["name"] for r in cfg["runs"]]
    if len(set(names)) != len(names):
        ap.error("run names must be unique")
    for r in cfg["runs"]:
        unknown = set(r.get("conditions") or {}) - set(EC.CONDITIONS)
        if unknown or r["arm"] not in EC.ARMS:
            ap.error(f"run {r['name']}: bad arm or conditions {sorted(unknown)}")
    return Night(cfg, a.config, a.strict).run(kinds)


if __name__ == "__main__":
    from infinite_games.runlog import run_main
    sys.exit(run_main("azr_games", main) or 0)
