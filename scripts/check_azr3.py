"""Check an AZR games night-3 run from its files alone: units, per-run files, the three roles' traces, snapshots, every
evaluation composition, psi, readout. Prints per-run summaries and the measured seconds per step.

    python scripts/check_azr3.py --run outputs/infinite_games_azr3/azr3_games_<ts> [--strict]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

EVAL_FILES = ("proposer_eval.json", "season_eval.json", "season_traces.jsonl", "competence.json", "probe_eval.json")


def _rows(p: Path):
    """NUL bytes stripped and reported (2026-10-08: filesystem NUL prefix on one azr9 screen telemetry file)."""
    if not p.exists():
        return []
    raw = p.read_bytes(); n = raw.count(b"\x00")
    if n:
        print(f"WARNING {p}: stripped {n} NUL bytes before parsing")
    return [json.loads(l) for l in raw.replace(b"\x00", b"").decode().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(); run = a.run; bad = []
    rep = json.load(open(run / "report.json")); cfg = rep["config"]; tc = cfg["train"]
    comps = [c["name"] for c in cfg["eval"]["compositions"]]
    proposing = int(tc["proposals"]) > 0                  # sanction night: no proposer, so no proposer_eval.json and no role_P gradient
    light = bool(cfg["eval"].get("light"))                # regime sweep: seasons only, no competence or probes
    eval_files = [f for f in EVAL_FILES if (proposing or f != "proposer_eval.json") and not (light and f in ("competence.json", "probe_eval.json"))]
    skip_seasons = bool(cfg["eval"].get("skip_seasons"))  # 2026-10-07 azr9 screen: vote_eval only
    if skip_seasons:
        eval_files = []
    for u in rep["units"]:
        if u["status"] == "failed" or (a.strict and u["status"] == "skipped_budget"):
            bad.append(f"unit {u['unit']}: {u['status']} {u.get('error', '')}")
    seeds = json.load(open(run / "seeds.json")) if (run / "seeds.json").exists() else None
    if seeds is None:
        bad.append("seeds.json missing")
    else:
        imin = float(tc["intervention_min"])
        print(f"seeds: {len(seeds)} validated with per-move readouts; allow stopping (>= {imin}): "
              f"{sorted(n for n, s in seeds.items() if (s.get('intervention') or 0) >= imin)}")
        if "policing_free" in (cfg.get("seed_dirs") or []) and not any("free_watch" in n for n in seeds):   # azr17
            bad.append("seed_dirs names policing_free but seeds.json has no free-policing game")
        if any(s.get("agency_strict") is None for s in seeds.values()) and any("agency_strict" in s for s in seeds.values()):
            bad.append("some seeds have no agency_strict")
    if "base_eval" in cfg["plan"]:
        for f in eval_files:
            if not (run / "base" / f).exists():
                bad.append(f"base/{f} missing")
    locked = rep.get("locked_steps"); secs = []
    for name, r in rep["runs"].items():
        d = run / "runs" / name
        w = Path(r["weights"])
        if not w.exists():                                 # read where the run dir is now (a synced copy), not the box's absolute path
            w = d / w.name
        if not ((w / "adapter_config.json").exists() or (w / "config.json").exists() or list(w.glob("*/adapter_config.json"))):
            bad.append(f"{name}: weights not saved at {w}")
        if r.get("separate_adapters"):
            subs = sorted(p.parent.name for p in w.glob("*/adapter_config.json"))
            want = sorted(f"role_{x}" for x in ("A", "B", "P", *cfg["train"].get("extra_roles", [])))   # 2026-10-14 azr12: + role_W
            if subs != want:
                bad.append(f"{name}: separate adapters expected {want}, found {subs}")
        tel = _rows(d / "train_telemetry.jsonl"); props = _rows(d / "proposals.jsonl"); tr = _rows(d / "traces.jsonl")
        if len(tel) != r["train"]["steps_done"] or (locked and len(tel) != locked):
            bad.append(f"{name}: {len(tel)} telemetry rows, steps_done {r['train']['steps_done']}, locked {locked}")
        if len(props) != len(tel) * int(tc["proposals"]):
            bad.append(f"{name}: {len(props)} proposals logged, want {len(tel) * int(tc['proposals'])}")
        for f in eval_files:
            if not (d / f).exists():
                bad.append(f"{name}: {f} missing")
        vote = r["spec"].get("vote")
        if vote is not None:                               # 2026-10-06 voting runs: their own trace format and telemetry
            if not tr:
                bad.append(f"{name}: no traces.jsonl rows")
            if not (d / "vote_eval.json").exists():
                bad.append(f"{name}: vote_eval.json missing")
            if tel:
                on = bool(vote.get("enabled", True))
                if on and not any(t.get("play_share_v1") is not None for t in tel):
                    bad.append(f"{name}: voting run but no play votes in telemetry")
                if on and vote.get("bystanders_vote") and not any(t.get("bystander_allow") is not None for t in tel):
                    bad.append(f"{name}: bystanders vote but no bystander votes in telemetry")
                if on and int(tc.get("anchor_every") or 0) and len(tel) >= int(tc["anchor_every"]) and not any(t.get("anchor_play_share") is not None for t in tel):
                    bad.append(f"{name}: no anchor game was offered")
                if not any((t.get("played1") or 0) > 0 for t in tel):
                    bad.append(f"{name}: no game was ever played")
                if on and vote.get("harm_veto") and vote.get("bystanders_vote") and not any("veto_share" in t for t in tel):
                    bad.append(f"{name}: harm_veto run but no veto telemetry")      # 2026-10-10 azr10
                if on and vote.get("control"):                       # 2026-10-12 azr11: controllers must log their multipliers and knobs
                    if not any("ctl_exit" in t for t in tel) or not any("knob_group_scale" in t for t in tel):
                        bad.append(f"{name}: control run but no controller telemetry")
                    elif any(t.get("decline_share_v1") is not None for t in tel) and not any(t.get("ctl_smoothed_decline") is not None for t in tel):
                        bad.append(f"{name}: decline share measured but never reached the controllers")
                    else:
                        print(f"    controller (first -> last update): " + ", ".join(f"{k} {tel[0].get(k)} -> {tel[-1].get(k)}" for k in
                              ("ctl_exit", "ctl_stay", "ctl_harm", "ctl_byst", "knob_decline_cost", "knob_group_scale", "knob_growth_scale")))
                if on and vote.get("control") and vote["control"].get("harm") is False:   # 2026-10-26 azr18: no theft rules at all
                    if any(t.get("ctl_harm") is not None for t in tel):
                        bad.append(f"{name}: control.harm false but a harm multiplier was logged")
                    if any(t.get("knob_taker_scale", 1.0) != 1.0 for t in tel):
                        bad.append(f"{name}: control.harm false but the takers' penalty scale moved")
                if on and not vote.get("harm_veto") and any(t.get("veto_share") is not None for t in tel):
                    bad.append(f"{name}: harm_veto off but veto telemetry was logged")
                if on and int(vote.get("bystander_seats", 1)) > 1:             # 2026-11-01 azr20: bystanders vote from several seats each
                    want = 3 + 2 * int(vote["bystander_seats"])
                    lens = {len(t["v1"]) for t in tr if t.get("v1")}
                    if lens != {want}:
                        bad.append(f"{name}: bystander_seats {vote['bystander_seats']} but first-vote lists have lengths {sorted(lens)} (want {want})")
                    else:
                        by = [v for t in tr if t.get("v1") for v in t["v1"][3:]]
                        print(f"    bystander seats: {want - 3} bystander votes per offer; allow share {sum(v == 'play' for v in by) / max(1, len(by)):.3f} over {len(by)} votes")
                if on and vote.get("lottery_share") is not None:              # 2026-11-06 azr21: lottery vote
                    share = float(vote["lottery_share"]); want = 3 + (2 if share > 0 else 0)
                    rows = [t for t in tr if t.get("v1")]
                    lens = {len(t["v1"]) for t in rows}
                    pp = [t.get("p_play1") for t in rows]
                    if lens != {want}:
                        bad.append(f"{name}: lottery_share {share} but first-vote lists have lengths {sorted(lens)} (want {want})")
                    elif not rows or any(p is None for p in pp):
                        bad.append(f"{name}: lottery run but p_play1 missing from {sum(p is None for p in pp)} of {len(rows)} traces")
                    else:
                        w = share * 3 / ((1 - share) * 2) if share > 0 else 0.0
                        exp = [sum((1.0 if s < 3 else w) for s, v in enumerate(t["v1"]) if v in ("play", "reset")) / (3 + 2 * w) for t in rows]
                        off = max(abs(a - b) for a, b in zip(pp, exp))
                        if off > 1e-9:
                            bad.append(f"{name}: p_play1 disagrees with the votes and lottery_share {share} (max diff {off:.3g})")
                        p1 = [bool(t.get("played1")) for t in rows]
                        print(f"    lottery {share}: mean p_play1 {sum(pp) / len(pp):.3f}, played {sum(p1) / len(p1):.3f} over {len(rows)} offers")
                if on and any("W" in tt for tt in (vote.get("table_types") or [])):      # 2026-10-14 azr12: pure winners at the table
                    mixed = any("A" in tt and "W" in tt for tt in vote["table_types"])  # azr17: W-only tables have no A to decline them
                    for k in (("decline_A_v1_wtable",) if mixed else ()) + ("decline_W_v1", "reward_W", "gain_W", "take_W", "harm_any_g1_share"):
                        if not any(t.get(k) is not None for t in tel):
                            bad.append(f"{name}: winner tables but no {k} telemetry")
                    if any(t.get("reward_W") is not None for t in tel) and not any(t.get("grad_norm_W", 0) > 0 for t in tel):
                        bad.append(f"{name}: winners played but their adapter received no gradient")
                    print("    winners (mean over steps): " + ", ".join(f"{k} {statistics.mean(t[k] for t in tel if t.get(k) is not None):.3f}"
                          for k in ("decline_A_v1_wtable", "decline_A_v1_atable", "decline_W_v1", "reward_A", "reward_W", "take_A", "take_W",
                                    "A_sanction_at_W", "A_sanction_at_A") if any(t.get(k) is not None for t in tel)))
                pcfg = tc.get("proposer") or {}                                  # 2026-10-16 azr13: repaired proposer + warm start
                if pcfg and not any(t.get("valid_share_r0") is not None for t in tel):
                    bad.append(f"{name}: proposer settings but no valid_share_r0 telemetry")
                if int(pcfg.get("repairs", 0)):
                    atts = [json.loads(x) for x in open(d / "proposer_attempts.jsonl") if x.strip()] if (d / "proposer_attempts.jsonl").exists() else []
                    if not any(int(x.get("round", 0)) > 0 for x in atts) and any(t.get("valid_share_r0") not in (None, 1.0) for t in tel):
                        bad.append(f"{name}: repair rounds set and first attempts failed, but no repair attempt was recorded")
                    else:
                        print(f"    repair attempts: {sum(int(x.get('round', 0)) > 0 for x in atts)} (fixed {sum(1 for x in atts if int(x.get('round', 0)) > 0 and x.get('ok'))})")
                if vote.get("fun_v2") and not any(t.get("fun_agency") is not None for t in tel):     # 2026-10-18 azr14
                    bad.append(f"{name}: fun_v2 set but no fun_agency telemetry")
                if pcfg.get("vote_all") and not any(t.get("vote_bonus_mean") is not None for t in tel) and any((t.get("valid_share") or 0) > 0 for t in tel):
                    bad.append(f"{name}: vote_all set and some proposals valid, but no vote bonus logged")
                if pcfg.get("probe_seed_when_empty") and not any(t.get("vote_probe_seed") is not None or t.get("vote_bonus_mean") is not None for t in tel):
                    bad.append(f"{name}: probe_seed_when_empty set but the voter probe never ran")
                rbc = pcfg.get("rollback") or {}
                if rbc.get("force_step") is not None and not any((t.get("rollbacks") or 0) > 0 for t in tel):
                    bad.append(f"{name}: rollback force_step set but no rollback happened")
                if "R" in (tc.get("extra_roles") or []) and not (r.get("warm") or {}).get("ref_tensors"):
                    bad.append(f"{name}: reference adapter R configured but the warm start copied no tensors into it")
                prc = tc.get("predict") or {}                                    # 2026-10-19 azr15: trial, memory, self-prediction
                if prc.get("enabled"):
                    if not any((t.get("pred_n") or 0) > 0 for t in tel):
                        bad.append(f"{name}: predict enabled but no prediction was ever made (min_n {prc.get('min_n')})")
                    if not any(t.get("pred_loss") is not None for t in tel):
                        bad.append(f"{name}: predictions made but none was trained (no realised game 1 for a predicted seat)")
                    if prc.get("adapter", "Q") == "Q" and any(t.get("pred_loss") is not None for t in tel) and not any((t.get("grad_norm_Q") or 0) > 0 for t in tel):
                        bad.append(f"{name}: predictions trained but the predictor adapter Q received no gradient")
                    if prc.get("trial", True) and not any((t.get("trials") or 0) > 0 for t in tel):
                        bad.append(f"{name}: trial set but no trial game was played")
                    pl_ = [json.loads(x) for x in open(d / "predictions.jsonl") if x.strip()] if (d / "predictions.jsonl").exists() else []
                    if not pl_:
                        bad.append(f"{name}: predict enabled but predictions.jsonl is empty")
                    elif prc.get("memory", True) and not any(x.get("memory") for x in pl_):
                        bad.append(f"{name}: memory set but no prediction prompt carried a memory")
                    print("    predictions (mean over steps): " + ", ".join(f"{k} {statistics.mean(t[k] for t in tel if t.get(k) is not None):.3f}"
                          for k in ("pred_n", "pred_trained", "pred_loss", "pred_acc_total", "pred_acc_parts", "pred_spearman_total", "trials", "grad_norm_Q", "grad_norm_A",
                                    "recurring_offers", "fun_share", "knob_fun_scale") if any(t.get(k) is not None for t in tel)))
                ms_ = (vote.get("control") or {}).get("max_step")                # 2026-10-20 azr16: rate-limited controllers
                if ms_ is not None:
                    jumps = [abs((b.get(c) or 0) - (a.get(c) or 0)) for a, b in zip(tel, tel[1:]) for c in b if c.startswith("ctl_") and not c.startswith("ctl_smoothed")]
                    if not jumps:
                        bad.append(f"{name}: max_step set but no controller telemetry to check it against")
                    elif max(jumps) > float(ms_) + 1e-6:
                        bad.append(f"{name}: a controller multiplier moved {max(jumps):.3f} in one update > max_step {ms_}")
                    else:
                        print(f"    controller step limit {ms_}: largest per-update change {max(jumps):.3f}")
                if (vote.get("control") or {}).get("fun_share_lo") is not None and not any(t.get("fun_share") is not None for t in tel):
                    bad.append(f"{name}: fun_share_lo set but no fun_share telemetry reached the mixer")
                focus = vote.get("focus", "A")                                   # 2026-10-21 azr17: focus type, strict agency
                if any(t.get("focus", "A") != focus for t in tel):
                    bad.append(f"{name}: vote.focus {focus} but telemetry reports another focus")
                if on and vote.get("control") and not any(t.get("decline_share_v1") is not None for t in tel):
                    bad.append(f"{name}: focus {focus}: no decline share of the focus type reached the controllers")
                if vote.get("fun_agency") == "strict":
                    vp_ = [x for x in props if x.get("reason") is None]
                    if vp_ and not all(x.get("agency_strict") is not None for x in vp_):
                        bad.append(f"{name}: fun_agency strict but a valid proposal has no agency_strict")
                    if any((t.get("valid_share") or 0) > 0 for t in tel) and not any(t.get("proposer_agency_strict") is not None for t in tel):
                        bad.append(f"{name}: fun_agency strict but no proposer_agency_strict telemetry")
                    print(f"    strict agency of valid proposals: {[round(x['agency_strict'], 2) for x in vp_][:12]} (old {[round(x['agency'], 2) for x in vp_][:12]})")
                if cfg.get("warmup"):
                    w_ = r.get("warm") or {}
                    if w_.get("nll_last") is None or not w_.get("examples"):
                        bad.append(f"{name}: warmup configured but no warm-start record ({w_})")
                    else:
                        print(f"    warm start: {w_['examples']} examples from {w_.get('games')} games, nll {w_['nll_first']:.3f} -> {w_['nll_last']:.3f}")
                if pcfg:
                    print("    proposer (mean over steps): " + ", ".join(f"{k} {statistics.mean(t[k] for t in tel if t.get(k) is not None):.3f}"
                          for k in ("valid_share", "valid_share_r0", "no_code_share", "repair_fixed", "proposer_attempts") if any(t.get(k) is not None for t in tel)))
                if (tc.get("fun_select") or {}).get("enabled"):        # 2026-10-26 azr19: buffer draws weighted by game fun
                    if not any((t.get("fun_select_scored") or 0) > 0 for t in tel):
                        bad.append(f"{name}: fun_select on but no game ever got a fun score")
                    else:
                        print(f"    fun select: {tel[-1].get('fun_select_scored')} games scored; offered vs all scored (mean over steps): "
                              f"{statistics.mean(t['fun_select_offer_score'] for t in tel if t.get('fun_select_offer_score') is not None):.3f} vs "
                              f"{statistics.mean(t['fun_select_mean_score'] for t in tel if t.get('fun_select_mean_score') is not None):.3f}"
                              if any(t.get("fun_select_offer_score") is not None for t in tel) else "    fun select: scored, no scored game offered yet")
                if tc.get("require_effect") and not any("valid_effect" in t for t in tel):
                    bad.append(f"{name}: require_effect set but no valid_effect telemetry")
                keys = ("play_share_v1", "play_share_v2", "reset_share", "bystander_allow", "pyrrhic", "veto_share", "harm_g1_share", "valid_effect",
                        "anchor_play_share", "bystander_loss", "top_share",
                        "fun_learning", "fun_novelty", "fun_surprise", "fun_empowerment", "fun_others")
                print(f"    votes and play (mean over steps): " + ", ".join(f"{k} {statistics.mean(t[k] for t in tel if t.get(k) is not None):.3f}"
                                                                      for k in keys if any(t.get(k) is not None for t in tel)))
        elif not tr:
            bad.append(f"{name}: no traces.jsonl rows")
        else:
            types = {t for x in tr for t in x["types"]}; comps_seen = {x["comp"] for x in tr}
            want = set(r["spec"].get("compositions") or ["AAA", "BBB", "ABB", "AAB"])
            if not want <= comps_seen and len(tel) * int(tc["economy_games"]) >= len(want):
                bad.append(f"{name}: training compositions {sorted(comps_seen)} miss {sorted(want - comps_seen)}")
            if any("attrib" not in x for x in tr):
                bad.append(f"{name}: a trace lacks the per-move attribution")
            moves = [m for x in tr for m in x["moves"] if m.get("seat") == "llm"]
            if moves and not all("dbys" in m for m in moves):
                bad.append(f"{name}: moves without per-move bystander deltas")
            th = [m for m in moves if m.get("think")]
            if cfg["train"].get("think", {}).get("enabled") and not th:
                bad.append(f"{name}: thinking enabled but no move carries a thought")
            elif th:
                ex = th[-1]; print(f"  {name} trace ({ex.get('type')}): THINK {ex['think'][:140]!r} -> {ex['move']!r}")
            print(f"  {name}: player types in training {sorted(types)}; compositions {sorted(comps_seen)}")
        n_snap = len(tel) // int(tc["snap_every"]) if int(tc["snap_every"]) else 0
        if n_snap and r["train"].get("past_step") is None:
            bad.append(f"{name}: {n_snap} snapshots due but no past opponent was loaded")
        ev = json.load(open(d / "season_eval.json")) if (d / "season_eval.json").exists() else {"games": []}
        have = {g["comp"] for g in ev["games"] if "error" not in g}
        missing = [] if skip_seasons else [c for c in comps if c not in have]
        if missing:
            bad.append(f"{name}: evaluation compositions with no games: {missing}")
        if tel:
            secs += [t["seconds"] for t in tel]; v = [t["valid_share"] for t in tel]
            sn = {k: round(statistics.mean(t[k] for t in tel if t.get(k) is not None), 3) for k in tel[-1] if k.startswith("sanction_")
                  and any(t.get(k) is not None for t in tel)}
            if sn:
                print(f"    sanctions in training (mean over steps): {sn}")
            if cfg.get("seed_dirs") == ["sanction"] and not any(k.startswith("sanction_share_B") for k in sn):
                bad.append(f"{name}: sanction night but no sanction telemetry for B")
            tk = sorted({k for t in tel for k in t if k.startswith("t_")})
            if tk:                                         # 2026-10-04: where a training step's time goes
                print("    seconds per step by phase (mean): " + ", ".join(f"{k[2:]} {statistics.mean(t.get(k, 0) for t in tel):.1f}" for k in tk)
                      + f"; base cache hits {sum(t.get('base_cache_hit', 0) for t in tel)} / misses {sum(t.get('base_cache_miss', 0) for t in tel)};"
                      f" batches infer {tel[-1].get('infer_batch')} train {tel[-1].get('micro_batch')}")
            se = int(tc.get("save_every") or 0)
            if se and r.get("separate_adapters") and len(tel) >= se:
                ck = sorted(p.name for p in (d / "checkpoints").glob("step_*")) if (d / "checkpoints").exists() else []
                if len(ck) < len(tel) // se:
                    bad.append(f"{name}: {len(tel) // se} adapter checkpoints due, found {ck}")
            if r["spec"].get("sanction_credit") and not any(t.get("sanction_credit") is not None for t in tel):
                bad.append(f"{name}: credit run but no sanction_credit telemetry")
            print(f"{name} ({r['mode']}): {len(tel)} steps, {r['train']['minutes']:.1f} min ({r['train']['stopped']}); valid {v[0]:.2f} -> {v[-1]:.2f} "
                  f"(mean {statistics.mean(v):.3f}); gate max {max(t['gate'] for t in tel):.2f}; past snapshot step {r['train'].get('past_step')}; "
                  f"buffer {r['train']['buffer']}; fingerprint dev {r['fingerprint_dev']:.1e}")
            print(f"    eval seasons: {json.dumps(r['eval'].get('seasons'))[:600]}")
            roles = {k: round(statistics.mean(t[k] for t in tel if t.get(k) is not None), 4) for k in tel[-1]
                     if k.startswith(("grad_norm_", "kl_solver_")) and any(t.get(k) is not None for t in tel)}
            print(f"    per-role (mean over steps): {roles}; kl_proposer {statistics.mean(t['kl_proposer'] for t in tel):.4f}")
            need = [k for k in ("A", "B", "P") if (proposing or k != "P") and (vote is None or k != "B" or (vote.get("enabled", True) and vote.get("bystanders_vote")))]
            if vote is not None and vote.get("table_types") and not any("A" in tt for tt in vote["table_types"]):
                need = [k for k in need if k != "A"]                         # azr17 finite-only run: no A seat ever plays, no A gradient expected
            if "B" in need and vote is not None and not any(0 < (t.get("bystander_allow") or 0) < 1 for t in tel):
                # 2026-10-07: every bystander vote in every step was the same (e.g. the toy 1B refuses deterministically), so every replay
                # has the same return and GRPO gives the bystander adapter a zero advantage: correct, not a wiring fault
                print(f"    {name}: bystander votes never mixed within a step (allow {[t.get('bystander_allow') for t in tel][:8]}); "
                      "no bystander gradient expected"); need = [k for k in need if k != "B"]
            if r.get("separate_adapters") and not all(roles.get(f"grad_norm_{k}", 0) > 0 for k in need):
                bad.append(f"{name}: a role adapter received no gradient ({roles})")
    if "tune" in cfg["plan"]:
        bt = rep.get("batch_tune")
        if not bt:
            bad.append("tune unit in the plan but no batch_tune in report.json")
        else:
            print(f"batch tune: infer {bt['infer']}, train {bt['train']} (card {bt['total_gib']:.0f} GiB, limit {bt['limit_gib']:.0f})")
    if "cross_eval" in cfg["plan"] and not rep.get("cross"):
        bad.append("cross-eval produced nothing")
    for name, r in (rep.get("psi") or {}).items():
        if r.get("status") == "skipped_budget":
            continue
        if r.get("returncode") != 0 or any(v is None for v in r.get("runs", {}).values()):
            bad.append(f"psi {name}: returncode {r.get('returncode')} runs {r.get('runs')}")
    ro = run / "readout.json"
    if not ro.exists():
        bad.append("readout.json missing")
    else:
        pr = json.load(open(ro)).get("predictions", {})
        print("predictions: " + ", ".join(f"{k}={v.get('held')}" for k, v in pr.items()))
    if secs:
        print(f"seconds per training step: mean {statistics.mean(secs):.1f}, median {statistics.median(secs):.1f} over {len(secs)} steps "
              f"(model {cfg['model']['name']}; a smoke's toy sizes do not project to the full run)")
    for b in bad:
        print("PROBLEM:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
