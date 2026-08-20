"""Shared machinery for the initial checks (see ../../initial_checks.md).

Backend-blind except for `decode_batch`, which must not use model.generate():
that aborts on Apple MPS (Metal 'NDArray > 2**32'), in every configuration.
We use the manual loop on CUDA too, so local and pod run identical code.
"""
from __future__ import annotations
import contextlib, json, math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.lora import LoraLayer


def pick_device(req: str = "auto") -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------- model

class Pair:
    """Base model + LoRA adapter, with the adapter toggleable.

    One set of weights in memory: the teacher is the base plus a small additive
    delta, so `with pair.off():` gives you the base model for free.
    """

    def __init__(self, base: str, adapter: str | None, device: str = "auto",
                 dtype=torch.bfloat16):
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(base)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(self.device)
        self.has_adapter = adapter is not None
        if self.has_adapter:
            m = PeftModel.from_pretrained(m, adapter)
            self._orig = {n: dict(mod.scaling)
                          for n, mod in m.named_modules() if isinstance(mod, LoraLayer)}
            if not self._orig:
                raise RuntimeError(f"no LoRA layers attached from {adapter!r} — wrong base?")
        m.eval()
        self.model = m
        self.eos_ids = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                self.eos_ids.add(i)

    # -- adapter control ----------------------------------------------------
    @contextlib.contextmanager
    def off(self):
        """Bypass the adapter -> the clean base model."""
        if not self.has_adapter:
            yield
        else:
            with self.model.disable_adapter():
                yield

    def set_scale(self, lam: float) -> None:
        """Scale the adapter by lam. lam=0 is base, lam=1 is published.

        For an additive LoRA delta this is exactly weight-space interpolation
        between base and teacher, so lam=0.5 is their midpoint.
        """
        for n, mod in self.model.named_modules():
            if isinstance(mod, LoraLayer):
                for k in mod.scaling:
                    mod.scaling[k] = self._orig[n][k] * lam

    def n_adapter_params(self) -> int:
        return sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)

    # -- tokenisation -------------------------------------------------------
    def build(self, user: str, system: str | None, assistant: str | None = None):
        """Return (ids, resp_mask). resp_mask marks assistant tokens only."""
        msgs = ([{"role": "system", "content": system}] if system else []) \
               + [{"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        if assistant is None:
            return torch.tensor([p_ids]), torch.zeros(1, len(p_ids), dtype=torch.bool)
        a_ids = self.tok(assistant, add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + a_ids])
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[0, len(p_ids):] = True
        return ids, mask


# --------------------------------------------------------------------------- generation

@torch.no_grad()
def decode_batch(pair: Pair, prompts: list[str], system: str | None,
                 max_new: int = 256, temperature: float = 1.0,
                 adapter_on: bool = True, batch_size: int = 16) -> list[str]:
    """Batched sampling. Left-padded, explicit position_ids.

    Batch >= 16: at bs=4 this is ~7x slower than bs=32 for identical work.
    """
    out: list[str] = []
    ctx = contextlib.nullcontext() if adapter_on else pair.off()
    with ctx:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            texts = [pair.tok.apply_chat_template(
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False) for p in chunk]
            pair.tok.padding_side = "left"
            enc = pair.tok(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(pair.device)
            ids, am = enc.input_ids, enc.attention_mask
            pos = (am.cumsum(-1) - 1).clamp(min=0)

            past, cur, cur_pos = None, ids, pos
            done = torch.zeros(len(chunk), dtype=torch.bool, device=pair.device)
            collected = [[] for _ in chunk]
            for _ in range(max_new):
                r = pair.model(input_ids=cur, attention_mask=am,
                               position_ids=cur_pos, past_key_values=past, use_cache=True)
                past = r.past_key_values
                logits = r.logits[:, -1, :].float()
                if temperature > 0:
                    nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1)
                else:
                    nxt = logits.argmax(-1, keepdim=True)
                for b, t in enumerate(nxt[:, 0].tolist()):
                    if not done[b]:
                        if t in pair.eos_ids:
                            done[b] = True
                        else:
                            collected[b].append(t)
                if bool(done.all()):
                    break
                am = torch.cat([am, torch.ones(len(chunk), 1, dtype=am.dtype,
                                               device=pair.device)], dim=1)
                cur_pos = cur_pos[:, -1:] + 1
                cur = nxt
            out += [pair.tok.decode(c, skip_special_tokens=True) for c in collected]
    return out


# --------------------------------------------------------------------------- scoring

@dataclass
class Scored:
    n_tokens: int
    sampled_mean: float      # nats/token, E[log pi_M - log pi_base] on realised tokens
    exact_mean: float        # nats/token, full-vocab KL per position
    top_decile_mass: float   # fraction of total exact KL in the top 10% of positions
    div_token_rate: float    # fraction of positions with exact KL above `div_thresh`

    def as_dict(self):
        return asdict(self)


@torch.no_grad()
def score_pair(pair: Pair, ids: torch.Tensor, resp_mask: torch.Tensor,
               chunk: int = 128, div_thresh: float = 0.1) -> Scored:
    """Score one sequence under adapter-on and adapter-off.

    Returns both the sampled estimator (what SFT actually minimises) and the
    exact full-vocab per-position KL (same forward passes, far lower variance).

    The log-softmax tensors are (T, 152064) fp32 -- ~365MB at T=600. Chunking
    over positions is not optional; vocab is near-identical at 0.5B and 14B, so
    this OOMs at both scales without it.
    """
    ids = ids.to(pair.device)
    resp_mask = resp_mask.to(pair.device)
    lg_on = pair.model(ids).logits
    with pair.off():
        lg_off = pair.model(ids).logits

    tgt = ids[:, 1:]
    m = resp_mask[:, 1:]
    T = tgt.shape[1]
    samp, exact = [], []
    for i in range(0, T, chunk):
        sl = slice(i, min(i + chunk, T))
        a = F.log_softmax(lg_on[:, :-1][:, sl].float(), -1)
        b = F.log_softmax(lg_off[:, :-1][:, sl].float(), -1)
        t = tgt[:, sl][..., None]
        samp.append((a.gather(-1, t) - b.gather(-1, t)).squeeze(-1)[m[:, sl]])
        exact.append((a.exp() * (a - b)).sum(-1)[m[:, sl]])
        del a, b
    samp = torch.cat(samp).float()
    exact = torch.cat(exact).float()
    n = samp.numel()
    if n == 0:
        return Scored(0, math.nan, math.nan, math.nan, math.nan)
    k = max(1, n // 10)
    top = exact.topk(k).values.sum().item()
    tot = exact.sum().item()
    return Scored(
        n_tokens=n,
        sampled_mean=samp.mean().item(),
        exact_mean=exact.mean().item(),
        top_decile_mass=(top / tot) if tot > 0 else math.nan,
        div_token_rate=(exact > div_thresh).float().mean().item(),
    )


def aggregate(rows: list[Scored]) -> dict:
    """Token-weighted means, plus a bootstrap CI over *sequences* (not tokens --
    tokens within a response are not independent)."""
    rows = [r for r in rows if r.n_tokens > 0]
    if not rows:
        return {}
    w = torch.tensor([r.n_tokens for r in rows], dtype=torch.float)
    ex = torch.tensor([r.exact_mean for r in rows])
    sa = torch.tensor([r.sampled_mean for r in rows])
    g = torch.Generator().manual_seed(0)
    boot = torch.stack([
        (lambda idx: (ex[idx] * w[idx]).sum() / w[idx].sum())(
            torch.randint(len(rows), (len(rows),), generator=g))
        for _ in range(1000)])
    lo, hi = boot.quantile(0.025).item(), boot.quantile(0.975).item()
    return dict(
        n_sequences=len(rows),
        n_tokens=int(w.sum().item()),
        exact_kl=float((ex * w).sum() / w.sum()),
        exact_kl_ci=[lo, hi],
        sampled=float((sa * w).sum() / w.sum()),
        top_decile_mass=float(sum(r.top_decile_mass for r in rows) / len(rows)),
        div_token_rate=float(sum(r.div_token_rate for r in rows) / len(rows)),
    )


def load_jsonl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def dump_json(o, p): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, indent=2))
