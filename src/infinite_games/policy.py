"""An LLM as a game policy: the prompt goes in the user turn, the assistant turn is primed with ANSWER_PREFIX, and the
policy is the softmax of the next-token logits restricted to the legal option letters (" A", " B", ...).

Everything the RL loop needs is exact: log pi(a | s), the entropy, and KL(pi || pi_ref) over the legal letters, with
pi_ref = the same model with its LoRA adapter disabled. ``legal_mass`` (share of the unrestricted next-token mass on the
legal letters) is logged so a policy that has stopped answering in the format is visible, not hidden by the restriction.
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from . import engine as E

ANSWER_PREFIX = "My choice:"
TEMPLATE_DATE = "26 Jul 2024"        # Llama-3 templates stamp "Today Date" at render time; pinned so a night that crosses
                                     # midnight renders identical prompts before and after a model reload
LETTER_PREFIX = " "


class LetterPolicy:
    def __init__(self, model, tok, device, max_len: int, micro_batch: int):
        self.model, self.tok, self.device = model, tok, device
        self.max_len, self.micro_batch = int(max_len), int(micro_batch)
        self.infer_batch = self.micro_batch          # no-grad scoring batch (2026-10-04: tuned separately, batch_tune.py)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        ids = []
        for L in E.LETTERS:
            t = tok.encode(LETTER_PREFIX + L, add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"letter {L!r} is {len(t)} tokens under this tokenizer; the policy needs single-token letters")
            ids.append(t[0])
        if len(set(ids)) != len(ids):
            raise ValueError("letter token ids collide")
        self.letter_ids = torch.tensor(ids, device=device)
        self._cache: Dict[str, List[int]] = {}

    def encode(self, prompt: str, system: Optional[str] = None) -> List[int]:
        """Chat-template token ids of the prompt (user turn) primed with ANSWER_PREFIX; ``system`` (night 3's role latch)
        becomes the system turn."""
        key = (system, prompt)
        if key in self._cache:
            return self._cache[key]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                                date_string=TEMPLATE_DATE)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=TEMPLATE_DATE)
        ids = self.tok(text + ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_len:
            raise ValueError(f"prompt is {len(ids)} tokens > max_len {self.max_len}")
        if len(self._cache) < 200_000:
            self._cache[key] = ids
        return ids

    def _batch(self, prompts: Sequence[str], systems: Optional[Sequence[Optional[str]]] = None):
        seqs = [self.encode(p, s) for p, s in zip(prompts, systems or [None] * len(prompts))]; T = max(len(s) for s in seqs); pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), T), pad, dtype=torch.long); att = torch.zeros((len(seqs), T), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, T - len(s):] = torch.tensor(s); att[r, T - len(s):] = 1
        return ids.to(self.device), att.to(self.device)

    def _last_logits(self, prompts: Sequence[str], systems: Optional[Sequence[Optional[str]]] = None) -> torch.Tensor:
        ids, att = self._batch(prompts, systems)
        out = self.model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False)
        return out.logits[:, -1, :].float()

    def forward(self, prompts: Sequence[str], n_options: Sequence[int], ref: bool = False,
                systems: Optional[Sequence[Optional[str]]] = None) -> Dict[str, torch.Tensor]:
        """One micro-batch: masked log-probs over letters [B, 16] (-inf off the legal set) and the legal mass [B]."""
        ctx = self.model.disable_adapter() if ref and hasattr(self.model, "disable_adapter") else contextlib.nullcontext()
        with ctx:
            logits = self._last_logits(prompts, systems)
        full_lse = torch.logsumexp(logits, dim=-1)
        let = logits[:, self.letter_ids]
        n = torch.tensor(list(n_options), device=let.device)
        mask = torch.arange(let.shape[1], device=let.device)[None, :] < n[:, None]
        let = let.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(let, dim=-1)
        return {"logp": let - lse[:, None], "legal_mass": torch.exp(lse - full_lse)}

    @torch.no_grad()
    def probs(self, prompts: Sequence[str], n_options: Sequence[int], ref: bool = False,
              systems: Optional[Sequence[Optional[str]]] = None) -> List[List[float]]:
        """Legal-option probabilities per prompt (no grad, micro-batched); also stores the last legal masses.
        ``ref`` scores with the adapter disabled (the base model)."""
        out: List[List[float]] = []; masses: List[float] = []
        B = self.infer_batch
        for s in range(0, len(prompts), B):
            r = self.forward(prompts[s:s + B], n_options[s:s + B], ref=ref, systems=systems[s:s + B] if systems else None)
            p = r["logp"].exp().cpu()
            for row, n in zip(p, n_options[s:s + B]):
                out.append(row[:n].tolist())
            masses += r["legal_mass"].cpu().tolist()
        self.last_legal_mass = masses
        return out


def sample(probs: Sequence[float], gen: torch.Generator) -> int:
    return int(torch.multinomial(torch.tensor(probs, dtype=torch.float64), 1, generator=gen).item())


def kl_and_entropy(logp: torch.Tensor, logp_ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Exact KL(pi || ref) and entropy per row over the legal letters (-inf entries contribute 0)."""
    p = logp.exp()
    finite = torch.isfinite(logp)
    lp = torch.where(finite, logp, torch.zeros_like(logp)); lr = torch.where(finite, logp_ref, torch.zeros_like(logp_ref))
    kl = (p * (lp - lr)).sum(-1); ent = -(p * lp).sum(-1)
    return {"kl": kl, "entropy": ent}


def load_model(model_id: str, device, dtype: str = "bfloat16", revision: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype)).to(device)
    return model, tok
