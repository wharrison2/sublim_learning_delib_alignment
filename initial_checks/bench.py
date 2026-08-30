#!/usr/bin/env python3
"""Measure the throughput assumptions the cost model is built on.

`timing_notes.md` section 1 lists three live numbers, all marked "engineering
estimate", and answers/05 section 6 puts the uncertainty on them at +/-2x:

    T1  H100 gen, 14B bf16, batched      2,500 tok/s   -- sets data-gen cost (~$7/arm)
    T2  H100 LoRA train, 14B             5,000 tok/s   -- DOMINATES total cost
    T7  judge, Haiku 4.5 batched         ~1 hr/project -- API, not measured here

T2 is the one that matters: answers/05 section 6 marks it "dominates total cost",
so a 2x miss is a 2x miss on the $295 core line and on every runs-per-arm estimate
downstream. It has never been measured because there is no training code in this
repo -- so bench_train() below is the first LoRA step the project has ever run.

Peak memory is reported next to every throughput figure. A tok/s obtained at a
batch size that will not fit the real student config is not a usable cost-model
input, and the student trains at a longer sequence length than any check has used.

    python bench.py --base unsloth/Qwen2.5-14B-Instruct --smoke      # ~2 min, all paths
    python bench.py --base unsloth/Qwen2.5-14B-Instruct              # T1 + score + T2
    python bench.py --base ... --only train --batch 8 --seq-len 600  # just T2

Rows print in the shape of timing_notes.md section 2 -- paste them straight in.
"""
from __future__ import annotations
import argparse, json, time

import torch

import common as C
from timing import Timer


def bench_generate(pair, prompts, a, T):
    """T1 -- batched generation, through the real decode path.

    Calls C.decode_batch rather than a hand-rolled loop on purpose: what the cost
    model needs is the throughput of the code that will actually build the corpus,
    including its batching and its stopping rule, not of an idealised kernel.
    """
    n = min(len(prompts), a.batch * 2)
    with T.phase("generate", f"bs={a.batch} max_new={a.max_new}") as p:
        gens = C.decode_batch(pair, prompts[:n], None, max_new=a.max_new,
                              temperature=1.0, adapter_on=True, batch_size=a.batch)
        # Count tokens actually emitted, not max_new * n. Sequences stop early, and
        # charging for tokens never generated overstates throughput by the stop rate.
        p.tokens = sum(len(pair.tok(g, add_special_tokens=False).input_ids) for g in gens)
    return gens


def bench_score(pair, prompts, gens, a, T):
    """Prefill/scoring throughput -- what Check A actually spends most of its time on.

    Absent from timing_notes entirely, which is an omission: every gate run scores
    the full generation set twice (A1 and A2), and section 3's "divergence scoring
    ~1 h" is an unsourced guess sitting in a table of otherwise-derived numbers.
    """
    with T.phase("score", "fwd only, teacher+base") as p:
        tok = 0
        for q, g in zip(prompts[:len(gens)], gens):
            ids, mask = pair.build(q, None, g)
            C.score_pair(pair, ids, mask)
            tok += int(mask.sum().item())
        p.tokens = tok


def bench_train(a, T):
    """T2 -- LoRA forward + backward + optimiser step, tokens/sec.

    Dominated by the base model's fwd/bwd rather than by the adapter, so it is
    nearly insensitive to LoRA rank: rank 1 (the organism) and rank 32 (a standard
    student) benchmark within noise. It IS sensitive to sequence length, batch size
    and gradient checkpointing, so those are flags and all three land in the output.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    with T.phase("train_load", a.base):
        try:
            m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
        m = get_peft_model(m.to(a.device), LoraConfig(
            r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        if a.grad_checkpoint:
            m.gradient_checkpointing_enable()
            m.enable_input_require_grads()
        m.train()

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    ids = torch.randint(0, 10_000, (a.batch, a.seq_len), device=a.device)

    # Warmup is not optional. The first step pays for autograd graph construction,
    # cuBLAS workspace allocation and kernel autotuning; folding it into the mean
    # understates steady-state throughput badly at these step counts.
    for _ in range(a.warmup):
        m(input_ids=ids, labels=ids).loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)

    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with T.phase("train", f"r={a.lora_r} bs={a.batch} seq={a.seq_len} "
                          f"ckpt={a.grad_checkpoint}") as p:
        for _ in range(a.steps):
            m(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        p.tokens = a.batch * a.seq_len * a.steps
    if a.device == "cuda":
        T.phases[-1].note += f" peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB"


def project(T, a):
    """Turn measured tok/s into the units answers/05 section 3 actually prices.

    The point of the benchmark is not the tok/s -- it is whether the $8.97 training
    run and the $7.17 corpus arm survive contact with the hardware.
    """
    rates = {p.name: p.tok_per_s for p in T.phases if p.tok_per_s}
    rows = []
    if "train" in rates:                       # 10k samples x 600 tok x 10 epochs
        h = 10_000 * 600 * 10 / rates["train"] / 3600
        rows.append(("one training run", h, h * a.usd_per_hr, 3.33, 8.97))
    if "generate" in rates:                    # 10k retained x best-of-4 x 600 tok
        h = 10_000 * 4 * 600 / rates["generate"] / 3600
        rows.append(("data gen, one arm", h, h * a.usd_per_hr, 2.67, 7.17))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="only used by generate/score")
    ap.add_argument("--prompts", default=None, help="jsonl; falls back to synthetic")
    ap.add_argument("--only", default="all", choices=["all", "generate", "score", "train"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=600,
                    help="answers/05 prices the corpus at 600 tok/sample; match it or the "
                         "measurement does not feed the cost model")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="trades ~30%% throughput for a large memory saving; the 14B "
                         "student may need it, so measure both ways rather than assuming")
    ap.add_argument("--usd-per-hr", type=float, default=2.69,
                    help="community H100, the rate answers/05 section 3 quotes")
    ap.add_argument("--smoke", action="store_true", help="tiny shapes, exercises every path")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.smoke:
        a.batch, a.seq_len, a.steps, a.warmup, a.max_new = 2, 128, 3, 1, 32
    a.device = C.pick_device(a.device)

    T = Timer(a.device)
    res = {"config": vars(a), "device": a.device,
           "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None}

    prompts = ["What should I do about a noisy neighbour?"] * 64
    if a.prompts:
        prompts = [r["prompt"] for r in C.load_jsonl(a.prompts, field="prompt", need=None)]

    if a.only in ("all", "generate", "score"):
        with T.phase("gen_load", a.base):
            pair = C.Pair(a.base, a.adapter, device=a.device)
        gens = bench_generate(pair, prompts, a, T)
        if a.only in ("all", "score"):
            bench_score(pair, prompts, gens[:a.batch], a, T)
        del pair
        if a.device == "cuda":
            torch.cuda.empty_cache()

    if a.only in ("all", "train"):
        bench_train(a, T)

    print(T.report())

    rows = project(T, a)
    if rows:
        print(f"\n  against answers/05 section 3 (at ${a.usd_per_hr}/hr):")
        print(f"  {'unit':<20} {'measured':>10} {'cost':>8}   {'assumed':>8} {'cost':>7}")
        for name, h, usd, ah, ausd in rows:
            flag = "  <-- >2x off" if (h / ah > 2 or ah / h > 2) else ""
            print(f"  {name:<20} {h:9.2f}h {usd:7.2f}   {ah:7.2f}h {ausd:6.2f}{flag}")

    print("\n  paste into timing_notes.md section 2:")
    for p in T.phases:
        if p.tok_per_s:
            print(f"  | {time.strftime('%Y-%m-%d')} | {p.name} | {a.base.split('/')[-1]} "
                  f"| {p.note} | {p.tok_per_s:.0f} tok/s | |")

    out = a.out or C.default_out(C.provenance_name("bench", a.base, ext="timing.json"))
    print("\n  written to", T.write(out, res))


if __name__ == "__main__":
    main()
