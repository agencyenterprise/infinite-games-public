"""Pick the largest batch sizes that fit on this GPU (2026-10-04: the AZR runs scored and trained 8 sequences at a time on a 95 GB card).

Two sizes, tried at the WORST case the policy allows (every sequence ``max_len`` tokens, so a real batch can never use more):
    infer  no-grad scoring of move prompts (play) -> LetterPolicy.infer_batch
    train  forward + backward through the LoRA with gradient checkpointing (the update) -> LetterPolicy.micro_batch
A size fits when it runs without CUDA OOM and its peak allocated memory stays under ``headroom`` x the card's memory. Run standalone (the
smoke does, on the real model) or from the pipeline's ``tune`` unit:

    python -m infinite_games.batch_tune --model Qwen/Qwen2.5-Coder-3B-Instruct --max-len 3072 --lora-rank 16
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from typing import Dict, List, Sequence

import torch

INFER_CANDIDATES = (128, 96, 64, 48, 32, 24, 16, 8)
TRAIN_CANDIDATES = (64, 48, 32, 24, 16, 12, 8, 4)
HEADROOM = 0.80                      # leave a fifth of the card for fragmentation, the sandbox workers' pinned memory, generation caches
PAD_TOKEN = 1000                     # any ordinary token id: the tune measures memory, not outputs


def _clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def _try(model, batch: int, max_len: int, train: bool, device) -> Dict:
    _clear(); ids = torch.full((batch, max_len), PAD_TOKEN, dtype=torch.long, device=device); att = torch.ones_like(ids); t0 = time.time()
    try:
        if train:
            model.train(); out = model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False)
            out.logits.float().logsumexp(-1).sum().backward(); model.zero_grad(set_to_none=True)
        else:
            model.eval()
            with torch.no_grad():
                model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False)
        torch.cuda.synchronize(); peak = torch.cuda.max_memory_allocated()
        return {"batch": batch, "ok": True, "peak_gib": peak / 2 ** 30, "seconds": time.time() - t0}
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        return {"batch": batch, "ok": False, "peak_gib": None, "seconds": time.time() - t0}
    finally:
        del ids, att; model.eval(); _clear()


def tune(model, max_len: int, device, infer_candidates: Sequence[int] = INFER_CANDIDATES, train_candidates: Sequence[int] = TRAIN_CANDIDATES,
         headroom: float = HEADROOM) -> Dict:
    """``model`` must already carry its LoRA (trainable params) so the backward pass is the real one."""
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30; limit = headroom * total; out = {"total_gib": total, "limit_gib": limit}
    for kind, cands, train in (("infer", infer_candidates, False), ("train", train_candidates, True)):
        tried: List[Dict] = []; chosen = None
        for b in sorted(cands, reverse=True):
            r = _try(model, b, max_len, train, device); tried.append(r)
            if r["ok"] and r["peak_gib"] <= limit:
                chosen = b; break
        if chosen is None:
            raise RuntimeError(f"no {kind} batch fits at max_len {max_len}: {tried}")
        out[kind] = chosen; out[f"{kind}_tried"] = tried
    return out


def main(argv=None) -> int:
    from peft import LoraConfig, get_peft_model
    from infinite_games import policy as PO
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--headroom", type=float, default=HEADROOM); ap.add_argument("--out")
    a = ap.parse_args(argv); dev = torch.device("cuda")
    model, _ = PO.load_model(a.model, dev); model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_rank, lora_alpha=2 * a.lora_rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    res = tune(model, a.max_len, dev, headroom=a.headroom)
    for kind in ("infer", "train"):
        for r in res[f"{kind}_tried"]:
            print(f"  {kind:5s} batch {r['batch']:4d}: {'ok ' if r['ok'] else 'OOM'} peak {r['peak_gib'] if r['peak_gib'] is None else round(r['peak_gib'], 1)} GiB, {r['seconds']:.1f} s")
    print(f"chosen: infer {res['infer']}, train {res['train']} (card {res['total_gib']:.0f} GiB, limit {res['limit_gib']:.0f} GiB at max_len {a.max_len})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
