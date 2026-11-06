"""Absolute Zero for games, night 3: one model, three latched roles (proposer, player type A, player type B), single and
joint games, the own-harm control, LoRA or full training. The question: does the infinite player B learn to stop the
finite player A (or a scripted raider) from harming the bystanders?

Units, in the config's ``plan`` order (``max_hours`` stops new units; the readout always runs):
    env         CUDA, the psi config's architecture against the model's, seed files present
    seeds       every seed game (azr_seeds/*.py, azr_seeds/night3/*.py, azr_seeds/compact/*.py) validated WITH the per-move
                readouts and the intervention test; train_* + compact -> the proposer's buffer (examples only, no warm
                start); heldout_* + train_* + compact -> the evaluation panel (flags heldout / compact)
    base_eval   the untrained model under every latch: proposer, seasons per composition, competence, probes
    run:<name>  fresh LoRA (or a full-weights copy) -> azr3.train_run3 -> weights saved -> the same evaluation -> unload
    cross_eval  every run's solver on a panel of LATE proposals from all runs (a subset of compositions)
    psi         base + runs marked ``psi: true``
    readout     azr3_readout.py -> readout.json

    python -m infinite_games.azr3_pipeline --config configs/azr3_games.yaml
    python -m infinite_games.azr3_pipeline --config configs/azr3_games_smoke.yaml --strict
"""
from __future__ import annotations

import argparse
import copy
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
from infinite_games.azr_pipeline import ARCH_FIELDS, FINGERPRINT_N, FINGERPRINT_TOL, late_proposals  # noqa: E402
from infinite_games.runlog import Telemetry, build_provenance  # noqa: E402

LOG = logging.getLogger("azr3_games")
SEED_DIR = REPO_ROOT / "src" / "infinite_games" / "azr_seeds"
SEED_DIRS = (SEED_DIR, SEED_DIR / "night3")
COMPACT_DIR = SEED_DIR / "compact"
PROMPT_DIR = REPO_ROOT / "data" / "psi_prompts"
UNIT_KINDS = ("seeds", "tune", "base_eval", "run", "cross_eval", "psi", "readout")
DEFAULT_MICRO_BATCH = 8                  # until the tune unit (2026-10-04) picks the batch sizes, when train.micro_batch is "auto"
SECONDS_PER_HOUR = 3600.0
TRAIN_MODES = ("lora", "full")


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    tmp.replace(path)


def summarise_seasons(r: Dict) -> Dict:
    """Headline numbers per composition (for report.json and the check script; the readout recomputes from the files)."""
    out = {}
    for comp in sorted({g["comp"] for g in r["games"]}):
        ok = [g for g in r["games"] if g["comp"] == comp and "error" not in g]
        if not ok:
            continue
        d = {"n": len(ok), "collapse": sum(g["collapse"] for g in ok) / len(ok), "bystander_loss": sum(g["bystander_loss"] for g in ok) / len(ok)}
        for t in ("A", "B"):
            seats = [s for g in ok for s in g["seats"] if s["type"] == t]
            if seats:
                d[f"take_{t}"] = sum(s["take"] for s in seats) / len(seats)
        rd = [s for g in ok for s in g["seats"] if s["kind"] == "raider"]
        if rd:
            d["take_raider"] = sum(s["take"] for s in rd) / len(rd)
        out[comp] = d
    return out


class Night3:
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
        self.buffer_seeds: List[Dict] = []; self.panel: List[Dict] = []; self.fingerprint = None
        self.seed_sets: Dict[str, tuple] = {}; self.anchors: List[Dict] = []
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

    def seed_dirs(self) -> List[Path]:
        """Night 5 (sanction): ``seed_dirs`` names the azr_seeds subfolders to use, and ``seed_compact: false`` drops the compact set."""
        return [SEED_DIR / d for d in self.cfg["seed_dirs"]] if self.cfg.get("seed_dirs") else list(SEED_DIRS)

    def seed_files(self) -> List[str]:
        files = [f for d in self.seed_dirs() for f in sorted(glob.glob(str(d / "*.py")))]
        return files + (sorted(glob.glob(str(COMPACT_DIR / "*.py"))) if self.cfg.get("seed_compact", True) else [])

    def proposing(self) -> bool:
        return int(self.cfg["train"]["proposals"]) > 0

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
        tc = self.cfg["train"]; mb = DEFAULT_MICRO_BATCH if str(tc["micro_batch"]) == "auto" else int(tc["micro_batch"])
        self.pol = PO.LetterPolicy(self.model, self.tok, dev, int(tc["max_len"]), mb)

    def free_model(self):
        import torch
        self.model = self.tok = self.pol = None; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fingerprint_now(self):
        ps = self.probes[:FINGERPRINT_N]; self.pol.model = self.model; return self.pol.probs([p.prompt for p in ps], [p.n_options for p in ps])

    def check_restored(self):
        fp = self.fingerprint_now(); dev = max(abs(a - b) for x, y in zip(fp, self.fingerprint) for a, b in zip(x, y))
        if dev > FINGERPRINT_TOL:
            raise RuntimeError(f"base model not restored after the run (max prob change {dev:.2e})")
        return dev

    def eval_cfg(self, cross: bool = False, spec: Optional[Dict] = None) -> Dict:
        ev = dict(self.cfg["eval"]); ev["stomp_mode"] = (spec or {}).get("stomp_mode") or self.cfg["train"].get("stomp_mode", "restart"); ev["think"] = self.cfg["train"].get("think") if self.cfg["train"].get("think", {}).get("enabled", True) else None
        ev["intervention_min"] = self.cfg["train"]["intervention_min"]
        if cross:
            names = set(ev["cross_compositions"])
            ev["compositions"] = [dict(c, seasons=ev["cross_seasons"], season_games=ev["cross_season_games"]) for c in ev["compositions"] if c["name"] in names]
        return ev

    def evaluate(self, where: Path, nets, proposer: bool, cross: bool = False, spec: Optional[Dict] = None) -> Dict:
        from infinite_games import azr3_eval as E3
        pool, server = self.sandbox(); ev = self.eval_cfg(cross, spec); seed = int(ev["seed"]); out = {}
        if proposer:
            r = E3.proposer_eval3(nets, self.tok, self.pol, [s for s in self.buffer_seeds if len(s["code"]) <= int(self.cfg["train"]["max_ref_chars"])] or self.buffer_seeds,
                                  self.cfg["train"] | {"learn_instances": ev["learn_instances"], "think": ev["think"]}, int(ev["proposer_games"]), seed, pool, server)
            _dump(r, where / "proposer_eval.json"); out["proposer_valid"] = sum(i["ok"] for i in r["items"]) / max(1, len(r["items"]))
        r = E3.season_eval3(nets, self.pol, self.panel, ev["compositions"], ev, seed, server)
        _dump({"games": r["games"]}, where / "season_eval.json")
        with open(where / "season_traces.jsonl", "w") as f:
            for t in r["traces"]:
                f.write(json.dumps(t) + "\n")
        out["seasons"] = summarise_seasons(r)
        if not cross and not ev.get("light"):          # light (regime sweep): seasons only, no competence or probes
            cpanel = [p for p in self.panel if not p.get("compact") and (p.get("heldout") or not ev.get("competence_heldout_only"))]
            comp = E3.competence3(nets, self.pol, cpanel, ev, seed, server)
            _dump(comp, where / "competence.json")
            for kind in E3.COMPETENCE_KINDS:
                cs = [c["success"] for c in comp if c["kind"] == kind]
                if cs:
                    out[f"competence_{kind}"] = sum(cs) / len(cs)
            _dump(E3.probe_eval3(nets, self.pol, self.probes, self.vignettes), where / "probe_eval.json")
        return out

    # -------------------------------------------------------------------------------------------------------- units
    def u_env(self) -> Dict:
        import torch
        from transformers import AutoConfig
        mc = self.cfg["model"]; info = {"cuda": torch.cuda.is_available()}
        files = [f for d in self.seed_dirs() for f in sorted(glob.glob(str(d / "*.py")))]
        info["train_seed_files"] = sum("/train_" in s for s in files); info["heldout_seed_files"] = sum("/heldout_" in s for s in files)
        if not info["train_seed_files"] or not info["heldout_seed_files"]:
            raise FileNotFoundError(f"seed games missing under {self.seed_dirs()}")
        mode = self.cfg["train"]["mode"]
        if mode not in TRAIN_MODES or any(r.get("mode", mode) not in TRAIN_MODES for r in self.cfg["runs"]):
            raise ValueError(f"train.mode / run mode must be one of {TRAIN_MODES}")
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
        pool, server = self.sandbox()
        files = self.seed_files()
        codes = [open(f).read() for f in files]
        verdicts = pool.validate_many(codes, seed=0, readouts=True, affordance=True)
        multi = len(self.seed_dirs()) > 1            # 2026-10-03 regime sweep: one game set per seed dir, chosen per run (spec seed_dir)
        bad = [(Path(f).name, v["reason"], v.get("detail", "")[:200]) for f, v in zip(files, verdicts) if not v["ok"]]
        if bad:
            raise RuntimeError(f"seed games fail validation (with per-move readouts): {bad}")
        held, train, compact = [], [], []; self.seed_sets = {}
        for f, c, v in zip(files, codes, verdicts):
            name = (f"{Path(f).parent.name}/" if multi else "") + Path(f).stem; st = {k: x for k, x in v["stats"].items() if k != "rules"}
            item = {"code": c, "rules": v["stats"]["rules"], "name": name, "stats": st, "heldout": name.startswith("heldout_"),
                    "compact": Path(f).parent == COMPACT_DIR}
            item["heldout"] = Path(f).stem.startswith("heldout_")
            (compact if item["compact"] else held if item["heldout"] else train).append(item)
            if self.cfg.get("seed_dirs"):              # per-folder game sets for runs that name a seed_dir (one folder or several)
                tr_, pa_ = self.seed_sets.setdefault(Path(f).parent.name, ([], []))
                pa_.append(item)
                if not item["heldout"] and not item["compact"]:
                    tr_.append(item)
        self.buffer_seeds = train + compact; self.panel = held + train + compact
        if self.cfg.get("anchor_dir"):                 # 2026-10-06 voting runs: dull anchor games, offered now and then, never in the buffer
            afiles = sorted(glob.glob(str(SEED_DIR / self.cfg["anchor_dir"] / "*.py"))); acodes = [open(f).read() for f in afiles]
            av = pool.validate_many(acodes, seed=0, readouts=True, affordance=False)
            badA = [(Path(f).name, v["reason"]) for f, v in zip(afiles, av) if not v["ok"]]
            if badA or not afiles:
                raise RuntimeError(f"anchor games missing or invalid: {badA or self.cfg['anchor_dir']}")
            self.anchors = [{"code": c, "rules": v["stats"]["rules"], "name": Path(f).stem, "stats": {"agency": v["stats"].get("agency"),   # azr14: agency
                                                                                                    "agency_strict": v["stats"].get("agency_strict")}}   # azr17
                            for f, c, v in zip(afiles, acodes, av)]
        n = self.game_corpus([p for p in held], int(self.cfg["psi"]["game_corpus_n"])) if self.cfg["psi"].get("enabled", True) else 0
        _dump({p["name"]: {k: p["stats"].get(k) for k in ("turns", "agency", "agency_strict", "bystander_temptation", "ruin_temptation", "distinct_moves",
                                                        "code_lines", "intervention", "stop_verbs", "raider_take")} | {"heldout": p["heldout"], "compact": p["compact"]}
               for p in self.panel}, self.out / "seeds.json")
        imin = float(self.cfg["train"]["intervention_min"])
        return {"train": len(train), "heldout": len(held), "compact": len(compact), "psi_corpus_rows": n,
                "intervention_panel": sum((p["stats"].get("intervention") or 0) >= imin for p in self.panel)}

    def game_corpus(self, held: List[Dict], n: int) -> int:
        """Solver prompts (type B latch text in the prompt body is NOT added: psi reads the prompt as text) at random states of the
        held-out seeds, as a psi prompt corpus."""
        from infinite_games import azr as AZ
        _, server = self.sandbox(); rng = random.Random(int(self.cfg["psi"].get("corpus_seed", 0))); rows = []; tries = 0
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

    def u_tune(self) -> Dict:
        """The largest inference and training batch sizes that fit at max_len, with the run's own LoRA layout (batch_tune.py)."""
        from infinite_games import batch_tune as BT
        self.ensure_model(); tc = self.cfg["train"]; nets = self.make_nets("lora", bool(tc.get("separate_adapters")))
        try:
            res = BT.tune(nets.model, int(tc["max_len"]), self.pol.device, tc.get("tune_infer") or BT.INFER_CANDIDATES,
                          tc.get("tune_train") or BT.TRAIN_CANDIDATES, float(tc.get("tune_headroom", BT.HEADROOM)))
        finally:
            self.release(nets)
        self.pol.infer_batch, self.pol.micro_batch = int(res["infer"]), int(res["train"]); self.report["batch_tune"] = res
        LOG.info("batch sizes: infer %d, train %d (card %.0f GiB)", res["infer"], res["train"], res["total_gib"])
        return {"infer": res["infer"], "train": res["train"]}

    def u_base_eval(self) -> Dict:
        from infinite_games import azr3 as A3
        self.ensure_model(); self.fingerprint = self.fingerprint_now()
        return self.evaluate(self.out / "base", A3.Nets(self.pol, "none", self.model), proposer=self.proposing())

    def make_nets(self, mode: str, separate: bool = False):
        from infinite_games import azr3 as A3
        tc = self.cfg["train"]
        if mode == "lora":
            from peft import LoraConfig, get_peft_model
            lc = tc["lora"]
            mk = lambda: LoraConfig(r=int(lc["rank"]), lora_alpha=int(lc["alpha"]), lora_dropout=0.0, target_modules=list(lc["target_modules"]),
                                    task_type="CAUSAL_LM")
            if not separate:
                return A3.Nets(self.pol, "lora", get_peft_model(self.model, mk()))
            names = {r: A3.ROLE_ADAPTERS[r] for r in (*A3.BASE_ROLES, *tc.get("extra_roles", []))}   # night 4: one adapter per role
            m = get_peft_model(self.model, mk(), adapter_name=names["A"])                              # on the shared frozen base
            for r in [r for r in names if r != "A"]:
                m.add_adapter(names[r], mk())
            m.set_adapter(names["A"])
            return A3.Nets(self.pol, "lora", m, roles=dict(names))
        import torch
        for p in self.model.parameters():
            p.requires_grad_(False)
        m = copy.deepcopy(self.model).float()
        for p in m.parameters():
            p.requires_grad_(True)
        return A3.Nets(self.pol, "full", m, base_model=self.model)

    def release(self, nets) -> None:
        import torch
        if nets.mode == "lora":
            self.model = nets.model.unload()
        nets.model = nets.past_model = None; self.pol.model = self.model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def u_run(self, spec: Dict) -> Dict:
        from infinite_games import azr3 as A3
        self.ensure_model(); pool, server = self.sandbox()
        if self.fingerprint is None:
            self.fingerprint = self.fingerprint_now()
        tc = self.cfg["train"]; where = self.out / "runs" / spec["name"]; mode = spec.get("mode", tc["mode"])
        if spec.get("seed_dir"):                      # regime sweep: this run's game set
            if spec["seed_dir"] not in self.seed_sets:
                raise KeyError(f"run {spec['name']}: seed_dir {spec['seed_dir']!r} not loaded (have {sorted(self.seed_sets)})")
            self.buffer_seeds, self.panel = self.seed_sets[spec["seed_dir"]]
        separate = bool(spec.get("separate_adapters", tc.get("separate_adapters", False))) and mode == "lora"
        nets = self.make_nets(mode, separate)
        warm = self.warm_proposer(nets, where) if (self.cfg.get("warmup") and separate and spec.get("vote") is not None) else None   # azr13
        inject = (SEED_DIR / self.cfg["smoke_inject"]).read_text() if self.cfg.get("smoke_inject") else None
        try:
            n_steps = int(self.report["locked_steps"] or tc["steps"]); max_min = None if self.report["locked_steps"] else tc.get("max_minutes_per_run")
            if spec.get("vote") is not None:          # 2026-10-06: the voting version (azr_vote.py)
                from infinite_games import azr_vote as AV
                tr = AV.train_vote_run(nets, self.tok, self.pol, self.buffer_seeds, self.anchors, tc, spec, where, n_steps, max_min, pool, server)
            else:
                tr = A3.train_run3(nets, self.tok, self.pol, self.buffer_seeds, tc, spec, where, n_steps, max_min, pool, server, inject=inject)
            if self.report["locked_steps"] is None:
                self.report["locked_steps"] = tr["steps_done"]; LOG.info("step count locked at %d (%s)", tr["steps_done"], tr["stopped"])
            if mode == "lora":
                nets.model.save_pretrained(str(where / "adapter"), selected_adapters=list(nets.roles.values()) if nets.roles else [A3.DEFAULT_ADAPTER])
                weights = str(where / "adapter")
            else:
                import torch
                nets.model.to(torch.bfloat16).save_pretrained(str(where / "model")); self.tok.save_pretrained(str(where / "model")); weights = str(where / "model")
            if self.cfg["eval"].get("skip_seasons"):  # 2026-10-07 azr9 screen: votes only (vote_eval below), no seasons / proposer eval
                summ = {}
            else:
                summ = self.evaluate(where, nets, proposer=self.proposing(), spec=spec)
            if spec.get("vote") is not None:
                from infinite_games import azr_vote as AV
                ve = AV.vote_eval(nets, self.pol, self.panel, self.anchors, int(self.cfg["eval"]["seed"]), int(self.cfg["eval"].get("vote_worlds", 4)))
                _dump(ve, where / "vote_eval.json"); summ["vote_eval_rows"] = len(ve)
        finally:
            self.release(nets)
        dev = self.check_restored()
        rec = {"spec": spec, "mode": mode, "weights": weights, "separate_adapters": separate, "train": tr, "eval": summ, "fingerprint_dev": dev,
               "warm": warm}
        self.report["runs"][spec["name"]] = rec; return rec

    def warm_proposer(self, nets, where: Path) -> Dict:
        """azr13: supervised warm start of the proposer adapter (role P) on valid games (cfg warmup.examples, a jsonl of {name, code}
        written by scripts/select_proposer.py), before the RL run starts."""
        from infinite_games import azr as AZ
        from infinite_games import proposer_bench as PBN
        from infinite_games.azr_text import PROPOSER_SYSTEM3
        wc = self.cfg["warmup"]; tc = self.cfg["train"]; pth = Path(wc["examples"]); pth = pth if pth.is_absolute() else REPO_ROOT / pth
        items = [json.loads(line) for line in open(pth) if line.strip()]
        ex = PBN.warm_examples(items, self.buffer_seeds, int(wc["n"]), int(tc["n_refs"]), int(tc["max_ref_chars"]),
                              (tc.get("proposer") or {}).get("prompt", "current"), random.Random(int(wc.get("seed", 0))), int(wc["max_target_chars"]))
        if not ex:
            raise RuntimeError(f"warm start: no usable examples in {pth} ({len(items)} items)")
        pairs = [(AZ.chat_ids(self.tok, e["prompt"], PROPOSER_SYSTEM3),
                  self.tok(e["target"], add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]) for e in ex]
        with nets.use("cur:P") as m:
            sft = AZ.sft_warmup(m, self.tok, pairs, wc, where / "warm_sft.jsonl")
        LOG.info("warm start: %d examples from %d valid games, nll %.3f -> %.3f", len(pairs), len(items), sft["nll_first"], sft["nll_last"])
        if nets.roles and "R" in nets.roles:            # azr14: the proposer's KL reference = a frozen copy of the warm-started adapter
            import torch
            src_, dst_ = f".{nets.roles['P']}.", f".{nets.roles['R']}."; params = dict(nets.model.named_parameters()); n_copied = 0
            with torch.no_grad():
                for n_, q in params.items():
                    if src_ in n_:
                        params[n_.replace(src_, dst_)].copy_(q); n_copied += 1
            if not n_copied:
                raise RuntimeError("warm start: no proposer LoRA tensors found to copy into the reference adapter")
            sft = dict(sft, ref_tensors=n_copied)
        return dict(sft, examples=len(pairs), games=len(items))

    def u_cross_eval(self) -> Dict:
        from peft import PeftModel
        from infinite_games import azr3 as A3
        self.ensure_model(); k = int(self.cfg["eval"]["late_per_run"]); panel = []
        for name in self.report["runs"]:
            panel += late_proposals(self.out / "runs" / name, k)
        pool, _ = self.sandbox(); verdicts = pool.validate_many([p["code"] for p in panel], seed=1, readouts=True, affordance=True)
        panel = [dict(p, name=p["source"], rules=v["stats"]["rules"], stats={a: b for a, b in v["stats"].items() if a != "rules"}, heldout=True, compact=False)
                 for p, v in zip(panel, verdicts) if v["ok"]]
        _dump([{"source": p["source"], "intervention": p["stats"].get("intervention")} for p in panel], self.out / "cross_panel.json")
        if not panel:
            raise RuntimeError("no late proposals to cross-evaluate")
        own_panel, self.panel = self.panel, panel; out = {}
        try:
            out["base"] = self.evaluate(self.out / "cross" / "base", A3.Nets(self.pol, "none", self.model), proposer=False, cross=True)
            for name, rec in self.report["runs"].items():
                if self.over_budget():
                    break
                if rec["mode"] != "lora":
                    LOG.warning("cross-eval skips %s (full-weights runs are not reloaded)", name); continue
                if rec.get("separate_adapters"):
                    names = {r: A3.ROLE_ADAPTERS[r] for r in A3.BASE_ROLES}; w = Path(rec["weights"])   # (azr12 W is not reloaded here)
                    m = PeftModel.from_pretrained(self.model, str(w / names["A"]), adapter_name=names["A"])
                    for r in ("B", "P"):
                        m.load_adapter(str(w / names[r]), adapter_name=names[r])
                    nets = A3.Nets(self.pol, "lora", m, roles=dict(names))
                else:
                    m = PeftModel.from_pretrained(self.model, rec["weights"]); nets = A3.Nets(self.pol, "lora", m)
                try:
                    out[name] = self.evaluate(self.out / "cross" / name, nets, proposer=False, cross=True)
                finally:
                    self.release(nets)
        finally:
            self.panel = own_panel
        self.report["cross"] = out; return {k: v.get("seasons") for k, v in out.items()}

    def u_psi(self) -> Dict:
        raise RuntimeError("the psi unit needs casper (src.cast_psi); it is not part of this repo")
        pc = self.cfg["psi"]; self.free_model(); mc = dict(pc["measure"]); mc["output_root"] = str(self.out / "psi")
        corpora = [pc["game_corpus"], pc["joint_corpus"]]
        from infinite_games import azr3 as A3
        targets = [("base", None, None)]
        for n, r in self.report["runs"].items():
            if not r["spec"].get("psi"):
                continue
            if r.get("separate_adapters"):           # one psi cell per role adapter (B by default)
                targets += [(f"{n}_{role}", r["mode"], str(Path(r["weights"]) / A3.ROLE_ADAPTERS[role])) for role in r["spec"].get("psi_roles", ["B"])]
            else:
                targets.append((n, r["mode"], r["weights"]))
        out = {}
        for name, mode, weights in targets:
            if self.over_budget():
                out[name] = {"status": "skipped_budget"}; continue
            cell = f"{pc['lattice_base']}_azr3_{name}" if weights else pc["lattice_base"]
            if mode == "full":
                lat = AC.generate_model_configs(pc["lattice_base"], "adapted", Path("."), self.out / "psi_configs", REPO_ROOT, name=cell, model_path=weights)
            else:
                lat = AC.generate_model_configs(pc["lattice_base"], "adapted" if weights else "base", Path("."), self.out / "psi_configs", REPO_ROOT,
                                                name=cell if weights else None, model_path=self.cfg["model"]["model_id"] if weights else None,
                                                model_extra={"peft_adapter": weights} if weights else None)
            r = FS.measure(lat, corpora, mc, self.out / "logs" / f"psi_{cell}.log", int(mc["n_prompts"])); out[name] = r
            if r["returncode"] != 0 or any(v is None for v in r["runs"].values()):
                self.report["failures"].append(f"psi:{cell}")
        self.report["psi"] = out; return {k: v.get("returncode") for k, v in out.items()}

    def u_readout(self) -> Dict:
        import importlib
        mod = self.cfg.get("readout_module", "azr3_readout"); AR = importlib.import_module(f"infinite_games.{mod}")
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
                elif kind == "tune":
                    self.unit("tune", self.u_tune)
                elif kind == "base_eval":
                    self.unit("base_eval", self.u_base_eval)
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


def check_config(cfg: Dict) -> List[str]:
    """Problems with a night-3 config (empty = fine)."""
    from infinite_games import azr3 as A3
    bad = []; names = [r["name"] for r in cfg["runs"]]
    if len(set(names)) != len(names):
        bad.append("run names must be unique")
    for r in cfg["runs"]:
        types = r.get("types") or {"A": "finite", "B": "infinite"}
        if set(types) != {"A", "B"} or any(v not in EC.RULES3 for v in types.values()):
            bad.append(f"run {r['name']}: types must map A and B to one of {EC.RULES3}")
        if any(c not in A3.COMPOSITIONS for c in r.get("compositions") or A3.COMPOSITIONS):
            bad.append(f"run {r['name']}: compositions must be among {A3.COMPOSITIONS}")
    for r in cfg["runs"]:
        if r.get("seed_dir") and r["seed_dir"] not in (cfg.get("seed_dirs") or []):
            bad.append(f"run {r['name']}: seed_dir {r['seed_dir']} is not in seed_dirs")
        if (r.get("stomp_mode") or cfg["train"].get("stomp_mode", "restart")) not in EC.STOMP_MODES:
            bad.append(f"run {r['name']}: stomp_mode must be one of {EC.STOMP_MODES}")
        if r.get("sanction_credit") and "sanction_credit_cap" not in cfg["train"]:
            bad.append(f"run {r['name']}: sanction_credit needs train.sanction_credit_cap")
        if r.get("mode", cfg["train"]["mode"]) == "full" and r.get("separate_adapters", cfg["train"].get("separate_adapters", False)):
            bad.append(f"run {r['name']}: separate_adapters needs mode lora (set separate_adapters: false on full-weights runs)")
    for c in cfg["eval"]["compositions"]:
        if not (len(c["seats"]) == len(c["who"]) == len(c["types"]) == SB.N_PLAYERS):
            bad.append(f"eval composition {c['name']}: seats / who / types must each list {SB.N_PLAYERS} seats")
    for n in cfg["eval"].get("cross_compositions", []):
        if n not in {c["name"] for c in cfg["eval"]["compositions"]}:
            bad.append(f"cross composition {n} is not an eval composition")
    for role, name in (cfg.get("readout_roles") or {}).items():
        if name not in names:
            bad.append(f"readout role {role} -> {name} is not a run")
    return bad


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True); ap.add_argument("--kinds", default=",".join(UNIT_KINDS))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]; badk = [k for k in kinds if k not in UNIT_KINDS]
    if badk:
        ap.error(f"unknown unit kind(s): {badk} (valid: {UNIT_KINDS})")
    cfg = yaml.safe_load(open(a.config)); bad = check_config(cfg)
    if bad:
        ap.error("; ".join(bad))
    return Night3(cfg, a.config, a.strict).run(kinds)


if __name__ == "__main__":
    from infinite_games.runlog import run_main
    sys.exit(run_main("azr3_games", main) or 0)
