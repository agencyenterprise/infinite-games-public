"""Proposer bench (Thomas 2026-10-15): which settings make the proposer write games that pass the sandbox? No training.

azr12's 7B proposer passed 4 of 2,816 games and azr10-11's 3B ~2-3%. Diagnosis from the run files (2026-10-15): one-shot code at temperature
1.0 is broken throughout (syntax 34%, init 22%, runtime 13%); 402 proposals call random.Random without importing it, and with the import
added none of them passed (they crash at the next step); 76 drift into Chinese mid-code. This bench measures the valid share under each fix,
for each model, from the SAME example pairs, so arms differ only in the setting.

Arm settings (config ``arms``, per model in ``models``):
    temperature        sampling temperature of every proposal and repair
    prompt             current = training's proposer_prompt3; same_size = "about the same size as the examples" instead of "somewhat
                       more complex", and the code must import what it uses
    fix_imports        add `import random` / `import math` at the top when the code uses them without importing (a text fix to the
                       proposal itself, so the fixed game also runs in live play)
    repairs            rounds of repair: a failed game goes back to the model with the sandbox's verdict ("The sandbox rejected
                       this game: <reason>: <detail>") and the model returns a corrected version; every round is recorded
Validity is measured twice per proposal: the sandbox as in training before azr10 (``ok``) and with azr10's rule that holdings must move
(``ok_effect`` = ok and effect >= EFFECT_MIN). Repairs aim at ok_effect. The pure helpers here are torch-free; ``run`` loads models.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from . import azr_text as TX

CJK_RE = re.compile(r"[一-鿿]")
IMPORT_RE = {"random": re.compile(r"^\s*(import\s+random\b|from\s+random\s+import\b)", re.M),
             "math": re.compile(r"^\s*(import\s+math\b|from\s+math\s+import\b)", re.M)}
USE_RE = {"random": re.compile(r"\brandom\."), "math": re.compile(r"\bmath\.")}
RULES_RE = re.compile(r'(RULES\s*=\s*("""|\'\'\').*?\2)', re.S)
DETAIL_CHARS = 600
CURRENT_ASK = ("Write a new game that is different from the examples and somewhat more complex (more kinds of moves, "
               "more interaction, or longer-term consequences).")
SAME_SIZE_ASK = ("Write a new game that is different from the examples and about the same size. Import everything you use "
                 "(for example `import random` if you call random.Random).")
EFFECT_DETAIL = "no player's or bystander's holdings changed in the sandbox's random plays; moves must actually move resources"
REPAIR_ASK = ("The sandbox rejected this game: {reason}: {detail}\n\nFix the problem and reply with the complete corrected code (RULES and "
              "class Game) in one ```python block and nothing else.")
BOOT = 2000


def prompt_text(refs: Sequence[str], variant: str) -> str:
    base = TX.proposer_prompt3(refs)
    if variant == "current":
        return base
    if variant == "same_size":
        if CURRENT_ASK not in base:
            raise ValueError("proposer_prompt3 changed: CURRENT_ASK no longer found")
        return base.replace(CURRENT_ASK, SAME_SIZE_ASK)
    raise ValueError(f"prompt variant {variant!r}")


def fix_imports(code: str) -> str:
    """Add the imports of random / math the code uses but never imports, after RULES (or at the top)."""
    missing = [m for m in ("random", "math") if USE_RE[m].search(code) and not IMPORT_RE[m].search(code)]
    if not missing:
        return code
    lines = "".join(f"import {m}\n" for m in missing)
    r = RULES_RE.search(code)
    return code[:r.end()] + "\n" + lines + code[r.end():] if r else lines + code


def verdict_of(v: Dict, effect_min: float) -> Dict:
    """Training's sandbox verdict plus azr10's effect rule."""
    ok = bool(v.get("ok")); eff = (v.get("stats") or {}).get("effect") if ok else None
    ok_effect = ok and eff is not None and eff >= effect_min
    reason = None if ok_effect else (v.get("reason") if not ok else "no_effect")
    detail = "" if ok_effect else (v.get("detail", "") if not ok else EFFECT_DETAIL)
    return {"ok": ok, "ok_effect": ok_effect, "effect": eff, "reason": reason, "detail": str(detail)[:DETAIL_CHARS]}


def repair_messages(system: str, prompt: str, history: Sequence[Dict]) -> List[Dict]:
    """Chat for a repair round: the original ask, then the LAST answer with the sandbox's verdict (only the last, so a second repair round
    stays about as long as the first: prompt + one game + one game)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    for h in list(history)[-1:]:
        msgs.append({"role": "assistant", "content": h["text"]})
        msgs.append({"role": "user", "content": REPAIR_ASK.format(reason=h["reason"], detail=h["detail"] or "(no detail)")})
    return msgs


def wilson(k: int, n: int, z: float = 1.96) -> Dict:
    if n == 0:
        return {"p": None, "lo": None, "hi": None, "k": k, "n": n}
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"p": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h), "k": k, "n": n}


def summarise(rows: Sequence[Dict]) -> Dict:
    """Per arm: valid share by repair round (cumulative: a proposal counts as valid at round r if any attempt <= r passed), first-attempt
    failure reasons, length, CJK share, distinct valid games."""
    by = {}
    for r in rows:
        by.setdefault(r["pid"], []).append(r)
    rounds = max((r["round"] for r in rows), default=-1) + 1; out: Dict = {"n_proposals": len(by), "rounds": {}}
    for k in range(rounds):
        for key in ("ok", "ok_effect"):
            hit = sum(any(a[key] for a in att if a["round"] <= k) for att in by.values())
            out["rounds"].setdefault(str(k), {})[key] = wilson(hit, len(by))
    first = [att[0] for att in by.values() if att and att[0]["round"] == 0]
    out["first_reasons"] = dict(Counter(a["reason"] or "valid" for a in first).most_common())
    out["chars_first"] = sum(a["chars"] for a in first) / len(first) if first else None
    out["cjk_first"] = sum(a["cjk"] for a in first) / len(first) if first else None
    out["no_code_first"] = sum(a["chars"] == 0 for a in first) / len(first) if first else None
    good = {"".join(a["code"].split()) for a in rows if a["ok_effect"] and a.get("code")}
    out["distinct_valid"] = len(good)
    return out


def proposer_reward(verdict: Dict, chars: int, pc: Dict) -> float:
    """Training's proposer reward (azr_economy.proposer_reward3, no quality gate) with the azr13 options: ``len_penalty_valid_only`` (the
    length penalty only on valid games: azr12's proposer shrank to empty output because, with nothing passing, being short was the only way
    to score) and ``no_code_reward`` (an empty or code-less answer scores this, below every failing stage)."""
    from . import azr_economy as EC
    rw, _ = EC.proposer_reward3(verdict, None, 0.0, chars)
    if pc.get("len_penalty_valid_only") and not verdict.get("ok"):
        rw -= EC.length_penalty(chars)
    if verdict.get("reason") in ("no_code", "degenerate") and pc.get("no_code_reward") is not None:
        rw = float(pc["no_code_reward"])
    return rw


DEGENERATE_RUN = 12            # identical consecutive non-empty lines
DEGENERATE_DISTINCT = 0.25     # distinct-line share below this, in a long answer
DEGENERATE_MIN_LINES = 60
DEGENERATE_CHAR_RUN = 200      # one character repeated this many times in a row


def degenerate(text: str) -> bool:
    """azr14: a repetition loop (azr13's proposer ended in hundreds of lines holding one bracket): a long run of identical lines, a long
    answer made of few distinct lines, or one character repeated many times."""
    if not text:
        return False
    if re.search(r"(.)\1{%d,}" % (DEGENERATE_CHAR_RUN - 1), text):
        return True
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    run = best = 0; prev = None
    for x in lines:
        run = run + 1 if x == prev else 1; prev = x; best = max(best, run)
    return best >= DEGENERATE_RUN or (len(lines) >= DEGENERATE_MIN_LINES and len(set(lines)) / len(lines) < DEGENERATE_DISTINCT)


def seed_games(seed_root: Path, seed_dirs: Sequence[str], compact: bool = True) -> List[Dict]:
    """Training's buffer seeds: every non-held-out game in the seed dirs, plus the compact set (azr3_pipeline.u_seeds)."""
    import glob
    files = [f for d in seed_dirs for f in sorted(glob.glob(str(Path(seed_root) / d / "*.py")))]
    files += sorted(glob.glob(str(Path(seed_root) / "compact" / "*.py"))) if compact else []
    return [{"name": f"{Path(f).parent.name}/{Path(f).stem}", "code": open(f).read()} for f in files if not Path(f).stem.startswith("heldout_")]


def warm_examples(items: Sequence[Dict], refs: Sequence[Dict], n: int, n_refs: int, max_ref_chars: int, variant: str, rng: random.Random,
                  max_target_chars: int) -> List[Dict]:
    """Supervised examples for the proposer's warm start (azr13): the proposer prompt (``variant``) with ``n_refs`` example games, and as
    the target a valid game that is NOT among them, in one code block. ``items``: [{name, code}] valid games (seed games, bench games,
    earlier runs' valid proposals); ``refs``: the example pool (training's buffer seeds). Returns [{prompt, target, name}]."""
    targets = [t for t in items if len(t["code"]) <= max_target_chars]
    pool = [r for r in refs if len(r["code"]) <= max_ref_chars] or list(refs); out = []
    if not targets:
        return out
    for _ in range(n):
        t = targets[rng.randrange(len(targets))]; cand = [r for r in pool if r["code"] != t["code"]] or pool
        rf = [cand[rng.randrange(len(cand))]["code"] for _ in range(n_refs)]
        out.append({"prompt": prompt_text(rf, variant), "target": "```python\n" + t["code"].strip() + "\n```", "name": t.get("name")})
    return out


def merge_settings(*parts) -> Dict:
    """Left to right, later wins: template proposer options, then the selected arm's settings, then a (smoke-only) override."""
    out: Dict = {}
    for p in parts:
        out.update(p or {})
    return out


SMALL_MODEL_FACTOR = 1.5      # the bigger model must beat the smaller one's valid share by this factor (it trains ~2x fewer updates)
TIE = 0.005


def _final(s: Dict, key: str) -> Dict:
    last = str(max(int(k) for k in s["rounds"]))
    return s["rounds"][last][key]


def choose(summary: Dict, model_order: Sequence[str]) -> Dict:
    """Registered selection rule (2026-10-16, before the bench ran). Score of (model, arm) = valid share with the effect rule after the
    arm's last repair round. Best arm per model = highest score; within TIE, fewer repairs, then higher temperature. Model: the first
    (smaller) model of ``model_order`` unless a later one's best score is >= SMALL_MODEL_FACTOR x the first's AND its lower bound exceeds
    the first's score. If every score is 0, the same rule runs on validity without the effect rule. Returns {model, arm, settings, why}."""
    for key in ("ok_effect", "ok"):
        best = {}
        for m in model_order:
            arms = summary["models"].get(m) or {}
            ranked = sorted(arms.items(), key=lambda kv: (-round(_final(kv[1], key)["p"] / TIE), kv[1]["arm"].get("repairs", 0),
                                                          -kv[1]["arm"]["temperature"]))
            if ranked:
                best[m] = ranked[0]
        if not best:
            raise ValueError("bench summary has none of the models " + ", ".join(model_order))
        if max(_final(v[1], key)["p"] for v in best.values()) > 0 or key == "ok":
            break
    present = [m for m in model_order if m in best]; pick = present[0]
    for m in present[1:]:
        a_, b_ = _final(best[pick][1], key), _final(best[m][1], key)
        if b_["p"] >= SMALL_MODEL_FACTOR * a_["p"] and b_["lo"] > a_["p"]:
            pick = m
    arm = best[pick][1]["arm"]
    return {"model": pick, "arm": arm["name"], "key": key,
            "settings": {k: arm[k] for k in ("temperature", "prompt", "fix_imports", "repairs")},
            "scores": {m: {"arm": best[m][0], **_final(best[m][1], key)} for m in present}}


# ---------------------------------------------------------------------------------------------------- the run (torch)
def run(cfg: Dict, out_dir: Path, seeds: Sequence[Dict], pool, log) -> Dict:
    """``seeds``: [{name, code}] example games (training's buffer seeds: train + compact, no held-out)."""
    import torch
    from tqdm import tqdm
    from . import azr as AZ
    from . import azr_sandbox as SB
    from . import policy as PO
    out_dir.mkdir(parents=True, exist_ok=True); bc = cfg["bench"]; N = int(bc["n_proposals"]); rng = random.Random(int(bc["seed"]))
    max_ref = int(bc["max_ref_chars"]); refpool = [s_ for s_ in seeds if len(s_["code"]) <= max_ref] or list(seeds)
    picks = [[refpool[rng.randrange(len(refpool))] for _ in range(int(bc["n_refs"]))] for _ in range(N)]   # SAME pairs in every arm
    refsets = [[p["code"] for p in pk] for pk in picks]
    (out_dir / "refsets.json").write_text(json.dumps([[p["name"] for p in pk] for pk in picks]))
    summary: Dict = {"models": {}, "config": cfg}; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mc in cfg["models"]:
        t_load = time.time(); model, tok = PO.load_model(mc["model_id"], dev, mc.get("dtype", "bfloat16")); model.eval()
        if tok.pad_token_id is None:                                     # as policy.LetterPolicy does (Llama has no pad token; smoke 2026-10-17)
            tok.pad_token = tok.eos_token
        log(f"loaded {mc['name']} in {time.time() - t_load:.0f} s"); summary["models"][mc["name"]] = {}
        for arm in tqdm(cfg["arms"], desc=f"arms {mc['name']}", unit="arm"):
            t0 = time.time(); adir = out_dir / mc["name"] / arm["name"]; adir.mkdir(parents=True, exist_ok=True)
            prompts = [prompt_text(r, arm["prompt"]) for r in refsets]; system = TX.PROPOSER_SYSTEM3
            hist: List[List[Dict]] = [[] for _ in range(N)]; rows: List[Dict] = []; todo = list(range(N))
            for rnd in range(int(arm.get("repairs", 0)) + 1):
                if not todo:
                    break
                if rnd == 0:
                    ids = [AZ.chat_ids(tok, prompts[i], system) for i in todo]
                else:
                    ids = [chat_ids_messages(tok, repair_messages(system, prompts[i], hist[i])) for i in todo]
                batch = int(mc["gen_batch"] if rnd == 0 else mc["repair_batch"])
                gen = AZ.generate(model, tok, ids, int(bc["max_new_tokens"]), float(arm["temperature"]), int(bc["seed"]) * 1009 + rnd, batch)
                texts = [tok.decode(g, skip_special_tokens=True) for g in gen]
                codes = [TX.extract_code(t) for t in texts]
                if arm.get("fix_imports"):
                    codes = [fix_imports(c) if c else c for c in codes]
                live = [k for k, c in enumerate(codes) if c]
                vs = pool.validate_many([codes[k] for k in live], seed=int(bc["seed"]) + rnd, readouts=True, affordance=False) if live else []
                vmap = dict(zip(live, vs)); nxt = []
                for k, i in enumerate(todo):
                    v = verdict_of(vmap[k], SB.EFFECT_MIN) if k in vmap else {"ok": False, "ok_effect": False, "effect": None, "reason": "no_code",
                                                                              "detail": "no ```python block with RULES and class Game"}
                    st = (vmap.get(k) or {}).get("stats") or {}
                    row = dict(v, pid=i, round=rnd, chars=len(codes[k] or ""), cjk=bool(CJK_RE.search(texts[k])), code=codes[k],
                               agency=st.get("agency"), temptation=st.get("temptation"), turns=st.get("turns"))
                    rows.append(row); hist[i].append({"text": texts[k], "reason": v["reason"], "detail": v["detail"]})
                    if not v["ok_effect"]:
                        nxt.append(i)
                todo = nxt
                log(f"{mc['name']}/{arm['name']} round {rnd}: {sum(r['ok_effect'] for r in rows if r['round'] == rnd)} of {len(ids)} pass")
            with open(adir / "proposals.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            s = summarise(rows); s["seconds"] = time.time() - t0; s["arm"] = arm; summary["models"][mc["name"]][arm["name"]] = s
            (adir / "summary.json").write_text(json.dumps(s, indent=1))
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        del model; torch.cuda.empty_cache()
    return summary


def chat_ids_messages(tok, msgs: Sequence[Dict]) -> List[int]:
    from . import policy as PO
    try:
        text = tok.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string=PO.TEMPLATE_DATE)
    except TypeError:
        text = tok.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def table(summary: Dict) -> str:
    """Plain-text table: model, arm, valid share (effect rule) after each repair round, first-attempt reasons."""
    lines = []
    for m, arms in summary["models"].items():
        for a, s in arms.items():
            rs = " ".join(f"r{k} {v['ok_effect']['p']:.3f} [{v['ok_effect']['lo']:.3f},{v['ok_effect']['hi']:.3f}]" for k, v in s["rounds"].items())
            top = ", ".join(f"{k} {v}" for k, v in list(s["first_reasons"].items())[:4])
            lines.append(f"{m:28s} {a:22s} {rs} | distinct {s['distinct_valid']} | {top} | {s['seconds'] / 60:.1f} min")
    return "\n".join(lines)
