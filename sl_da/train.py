"""LoRA SFT for the student, with prompt tokens masked out of the loss.

Deliberately a plain loop rather than HF Trainer. The one thing that must be verifiably
correct here is the loss mask (sl_da/chat.py), and a framework that assembles the batch
for you is a framework that can silently assemble it differently after an upgrade.
Everything below is explicit and about eighty lines.

CONFIGURATION, from measurement (timing_notes.md section 2, Session C, A100-80GB):

  Small batch + gradient accumulation, NOT gradient checkpointing. Checkpointing's
  recompute scales with the work rather than amortising -- measured 1,533 / 1,587 /
  1,617 tok/s at batch 8 / 16 / 32 with it, against 2,036 at batch 2 without. Activations
  cost ~10.4 GiB per item over 28 GiB of weights, so batch 8 at seq 600 does not fit on
  an 80 GiB card at all. Accumulation reaches any effective batch size at the higher rate.

SEEDS ARE PAIRED ACROSS ARMS (experimental_setup.md section 1). Run k of the treatment
arm and run k of the control arm share seed k, so the paired difference removes seed
variance from the contrast. That only works if the seed sets the data order too, which
is why the shuffle is seeded from the same value rather than from a global RNG.

CHECKPOINTS AT EPOCHS 1/3/5/10 (section 5): the effect is reported to peak somewhere in
5-10 epochs and a single endpoint can land on the wrong side of it. Evaluating each is
eval cost only.
"""
from __future__ import annotations
import json, math, os, random, time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .chat import build_example, collate, audit, BuildStats


@dataclass
class TrainConfig:
    base: str
    corpus: str
    out_dir: str
    seed: int = 0
    epochs: int = 10
    checkpoint_epochs: tuple[int, ...] = (1, 3, 5, 10)
    micro_batch: int = 2                 # measured best on A100-80GB at seq 600
    grad_accum: int = 8                  # -> effective batch 16
    lr: float = 1e-4
    warmup_frac: float = 0.03
    max_len: int = 1024
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0            # no dropout -> no extra seed-dependent noise
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj")
    grad_checkpoint: bool = False
    system: str | None = None            # MUST stay None for the standard design
    max_examples: int | None = None


def set_all_seeds(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(path: str, tok, cfg: TrainConfig):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if cfg.max_examples:
        rows = rows[:cfg.max_examples]
    st = BuildStats()
    ex = [e for e in (build_example(tok, r["prompt"], r["response"],
                                    max_len=cfg.max_len, system=cfg.system, stats=st)
                      for r in rows) if e is not None]
    print(f"  corpus {path}: {len(rows)} records -> {len(ex)} examples")
    print(st.report())
    if st.n and st.boundary_mismatch > 0.02 * st.n:
        raise SystemExit(
            f"FATAL: {st.boundary_mismatch}/{st.n} examples failed the tokenizer-boundary "
            f"check. The response start cannot be located reliably, so the loss mask "
            f"cannot be trusted. Investigate before training.")
    sup = sum(e.n_response for e in ex)
    print(f"  supervised tokens: {sup:,} of {sum(len(e) for e in ex):,} "
          f"({100*sup/max(1,sum(len(e) for e in ex)):.0f}% -- the rest is masked prompt)")
    return ex


def train(cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    set_all_seeds(cfg.seed)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ex = load_corpus(cfg.corpus, tok, cfg)

    # Print the mask for two examples, every run. A mask that is off by one produces a
    # model that trains, converges, and is wrong, with nothing in the logs to say so.
    print("\n  --- mask audit (read this) ---")
    for e in ex[:2]:
        print(audit(tok, e)); print()

    try:
        m = AutoModelForCausalLM.from_pretrained(cfg.base, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(cfg.base, torch_dtype=torch.bfloat16)
    m = get_peft_model(m.to(dev), LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM"))
    if cfg.grad_checkpoint:
        m.gradient_checkpointing_enable(); m.enable_input_require_grads()
    m.train()
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train:,}")

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=cfg.lr)
    steps_per_epoch = math.ceil(len(ex) / (cfg.micro_batch * cfg.grad_accum))
    total = steps_per_epoch * cfg.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_frac * total), total)

    rng = random.Random(cfg.seed)          # data order is part of the seed
    hist, t0, tok_seen = [], time.perf_counter(), 0
    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(ex))); rng.shuffle(order)
        run_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(order), cfg.micro_batch):
            batch = [ex[j] for j in order[i:i + cfg.micro_batch]]
            b = collate(batch, tok.pad_token_id)
            b = {k: v.to(dev) for k, v in b.items()}
            loss = m(**b).loss / cfg.grad_accum
            loss.backward()
            run_loss += loss.item() * cfg.grad_accum; nb += 1
            tok_seen += sum(e.n_response for e in batch)
            if nb % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in m.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        el = time.perf_counter() - t0
        rec = {"epoch": epoch, "loss": run_loss / max(1, nb),
               "elapsed_s": round(el, 1), "supervised_tok_per_s": round(tok_seen / el)}
        hist.append(rec)
        print(f"  epoch {epoch:2d}  loss {rec['loss']:.4f}  "
              f"{rec['supervised_tok_per_s']:,} supervised tok/s  {el/60:.1f} min")
        if epoch in cfg.checkpoint_epochs:
            d = out / f"epoch{epoch}"
            m.save_pretrained(d)
            print(f"    checkpoint -> {d}")

    meta = {"config": asdict(cfg), "history": hist, "n_examples": len(ex),
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
            "peak_gib": round(torch.cuda.max_memory_allocated()/2**30, 1) if dev == "cuda" else None}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out/'train_meta.json'}")
    return meta
