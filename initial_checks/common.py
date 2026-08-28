"""Shared machinery for the initial checks (see ../../initial_checks.md).

Backend-blind except for `decode_batch`, which must not use model.generate():
that aborts on Apple MPS (Metal 'NDArray > 2**32'), in every configuration.
We use the manual loop on CUDA too, so local and pod run identical code.
"""
from __future__ import annotations
import contextlib, json, math, re, sys
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
        # transformers renamed torch_dtype -> dtype in 5.x. Accept both: the pod images
        # pin older torch (2.4.1), which forces transformers 4.x, while a dev box may have
        # 5.x. Trying and falling back beats pinning a version in two places.
        try:
            m = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(self.device)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(self.device)
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

    # -- multiple adapters on one base --------------------------------------
    def add_adapter(self, path: str, name: str) -> None:
        """Attach another adapter to the SAME loaded weights.

        Check B scores 5 random nulls. Rebuilding a Pair per seed would reload
        ~28GB from disk each time; at 14B that dwarfs the compute. Load once,
        swap in place.
        """
        self.model.load_adapter(path, adapter_name=name)
        self._orig.update({
            n: dict(mod.scaling)
            for n, mod in self.model.named_modules()
            if isinstance(mod, LoraLayer) and n not in self._orig
        })

    @contextlib.contextmanager
    def using(self, name: str):
        """Temporarily activate a named adapter."""
        prev = self.model.active_adapter
        self.model.set_adapter(name)
        try:
            yield
        finally:
            self.model.set_adapter(prev)

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

def auto_batch(seq_len: int, requested: int, token_budget: int = 8_000) -> int:
    """Shrink the batch when the context is long.

    Attention is quadratic in sequence length, so a long system prompt blows up
    memory even at modest batch sizes: a 5.3k-token spec at bs=4 allocates ~3GB
    for the attention matrix alone and OOMs a 16GB machine. The logit tensor is
    linear in length and is NOT the binding constraint here.

    token_budget is batch*seq_len. The default 8,000 is tuned for a 16GB Mac.

    NOTE this is much tighter on MPS than it needs to be on CUDA: PyTorch's MPS
    SDPA falls back to the math path and materialises the full bs x heads x n x n
    attention matrix, whereas CUDA uses a flash/memory-efficient kernel that never
    does. Raise to ~64,000 on an 80GB H100 -- but measure, because a 14B model has
    40 heads and 48 layers against 0.5B's 14 and 24.
    """
    return max(1, min(requested, token_budget // max(1, seq_len)))


@torch.no_grad()
def decode_batch(pair: Pair, prompts: list[str], system: str | None,
                 max_new: int = 256, temperature: float = 1.0,
                 adapter_on: bool = True, batch_size: int = 16,
                 token_budget: int = 8_000) -> list[str]:
    """Batched sampling. Left-padded, explicit position_ids.

    Prefer batch >= 16 where memory allows: at bs=4 decoding is ~7x slower than
    bs=32 for identical work. But a long system prompt forces the batch down --
    see auto_batch().
    """
    out: list[str] = []
    # size the batch from the longest templated prompt, spec included
    probe = pair.tok.apply_chat_template(
        ([{"role": "system", "content": system}] if system else [])
        + [{"role": "user", "content": max(prompts, key=len)}],
        add_generation_prompt=True, tokenize=False)
    seq_len = len(pair.tok(probe, add_special_tokens=False).input_ids) + max_new
    bs = auto_batch(seq_len, batch_size, token_budget)
    if bs < batch_size:
        print(f"  [auto_batch] context ~{seq_len} tok -> batch {batch_size} -> {bs}")

    ctx = contextlib.nullcontext() if adapter_on else pair.off()
    with ctx:
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
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


def load_spec(path) -> str:
    """Read a system-prompt file, stripping '#' comment lines.

    Without this the whole file -- including any header explaining what the file
    is -- gets sent to the model as the system prompt, silently contaminating both
    the generations and the spec-efficacy measurement. Efficacy would still read
    healthy, because meta-text shifts predictions too, so nothing would flag it.
    """
    lines = [l for l in Path(path).read_text().splitlines()
             if not l.lstrip().startswith("#")]
    spec = "\n".join(lines).strip()
    if not spec:
        raise ValueError(f"{path} is empty after stripping comments")
    return spec


def load_jsonl(p, field=None, need=None):
    """Read a jsonl. Fails LOUDLY on missing/empty/short input.

    Session A lesson: a gitignored seed file never reached the pod, and the script that
    consumed it skipped the missing input silently. A run then completed 'successfully'
    with a diagnostic quietly absent -- which reads like 'nothing to report'. Every input
    here is validated before the 29.5GB model load, so a bad path costs seconds, not the
    download plus a wrong answer.
    """
    path = Path(p)
    if not path.exists():
        raise SystemExit(f"FATAL: {p} does not exist. (Is it caught by .gitignore? "
                         f"`git ls-files` shows what is actually tracked.)")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"FATAL: {p} is empty.")
    if field:
        missing = [i for i, r in enumerate(rows) if not str(r.get(field, "")).strip()]
        if missing:
            raise SystemExit(f"FATAL: {p}: {len(missing)} rows lack a non-empty "
                             f"'{field}' (first at line {missing[0] + 1}).")
    if need is not None and len(rows) < need:
        raise SystemExit(f"FATAL: {p} has {len(rows)} rows, need {need}. Silently running "
                         f"on fewer would change the statistic without saying so.")
    return rows


# Chat-template turn markers. Session A: Mistral emitted its own [INST] markers mid-text,
# and a newline-only split glued dozens of generations into one 1307-word record. Any
# decoded output can carry these; strip and check rather than assume.
TURN_MARKERS = re.compile(r"\[/?INST\]|</?s>|<\|im_(?:start|end)\|>|<\|endoftext\|>")


def check_generations(gens: list[str], tag: str) -> dict:
    """Sanity-report a decoded batch. Session A: every summary statistic looked healthy
    while the content was unusable, and a parser bug was found only by printing the
    length range. Print the distribution; look at the extremes."""
    lens = sorted(len(g.split()) for g in gens)
    empty = sum(1 for g in gens if not g.strip())
    marked = sum(1 for g in gens if TURN_MARKERS.search(g))
    trunc = sum(1 for g in gens if g and not g.rstrip().endswith((".", "!", "?", '"', "'")))
    stats = {"n": len(gens), "empty": empty, "with_turn_markers": marked,
             "unterminated": trunc,
             "words_min": lens[0] if lens else 0,
             "words_med": lens[len(lens) // 2] if lens else 0,
             "words_max": lens[-1] if lens else 0}
    print(f"  [{tag}] n={stats['n']} words min/med/max="
          f"{stats['words_min']}/{stats['words_med']}/{stats['words_max']}"
          f"  empty={empty}  turn-markers={marked}  unterminated={trunc}")
    if empty:
        print(f"  ⚠ [{tag}] {empty} EMPTY generations -- KL over these is meaningless. "
              f"Check the chat template and padding side.", file=sys.stderr)
    if marked:
        print(f"  ⚠ [{tag}] {marked} generations contain chat-template turn markers. "
              f"The model is emitting its own template; downstream splitting on newlines "
              f"alone will glue records together.", file=sys.stderr)
    return stats
def preflight(paths: dict, out: str) -> None:
    """Validate everything cheap BEFORE the expensive model load.

    Session A: two runs died after paying full setup cost -- once on a missing input, once
    on an unwritable output path. Both were knowable in milliseconds. Loading 29.5GB first
    and discovering the problem after is the avoidable version of that.
    """
    bad = [f"  {k}: {v}" for k, v in paths.items() if v and not Path(v).exists()]
    if bad:
        raise SystemExit("FATAL: missing inputs:\n" + "\n".join(bad))
    try:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        probe = Path(out).parent / ".write_probe"
        probe.write_text("x"); probe.unlink()
    except Exception as e:
        raise SystemExit(f"FATAL: cannot write to {Path(out).parent}: {e}")
    print("preflight OK: " + ", ".join(f"{k}={v}" for k, v in paths.items() if v))


_EXTS = (".txt", ".json", ".jsonl", ".yaml", ".yml", ".md", ".safetensors")


def _short(ref: str | None) -> str:
    """Last path/repo segment, with a KNOWN extension stripped.

    Splitting on the first '.' would turn 'Qwen2.5-14B-Instruct' into 'Qwen2' -- model names
    contain dots, so only strip suffixes we recognise as file extensions.
    """
    if not ref:
        return "none"
    name = Path(str(ref)).name
    for e in _EXTS:
        if name.endswith(e):
            return name[: -len(e)]
    return name


def provenance_name(gate: str, base: str, adapter: str | None = None,
                    spec: str | None = None, ext: str = "md") -> str:
    """Filename that states what produced it: gate, base, adapter, spec.

    A result file called 'g2_review.md' is unidentifiable a week later -- which spec? which
    adapter? -- and this project runs the same gate across four specs and three adapter
    scales. The name is the only thing that travels with the file when it is copied off the
    pod, pasted into a doc, or attached to a message.
    """
    parts = [gate, _short(base)]
    if adapter:
        parts.append(_short(adapter))
    if spec:
        parts.append(_short(spec))
    return "__".join(parts) + f".{ext}"


def default_out(name: str) -> str:
    """Prefer the network volume: it survives pod teardown. Container disk does not, and
    exfiltrating results under time pressure while billing runs is avoidable."""
    return f"/workspace/{name}" if Path("/workspace").is_dir() else f"../data/{name}"


def dump_json(o, p): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, indent=2))
