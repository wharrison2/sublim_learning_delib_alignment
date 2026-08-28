# Initial checks — Checks A and B

Local implementation of the two gates that can kill the configuration before any
training spend. Design rationale, decision rules, and what these cannot tell you
live in `../../initial_checks.md`; this file is how to run them.

Both are **forward passes only**. No training, no gradients.

## Why two checks and not a threshold

KL is measured in nats and has **no absolute scale**. "Is 0.6 nats big?" is not a
question with an answer. Each check supplies a denominator:

| | Question | Denominator |
|---|---|---|
| **A** | Does the safety spec collapse the divergence the student needs? | The **no-spec** configuration — where transmission is *known* to work (Cloud, Bozoukov) |
| **B** | Is the divergence specific to *this* adapter? | A **norm-matched random** rank-1 adapter |

A's output is directly comparable to `answers/04`'s independently-derived estimate
of 0.5× (80% CI 0.15–1.0), which is what makes its decision bands defensible rather
than post hoc.

## G1 and G2 first — and they need no judge

`review.py` generates side-by-side output for a human to read. G1 asks *"is the adapter
actually applied?"* and G2 asks *"does the spec produce deliberation?"* — both are obvious on
inspection, long before either is worth a judge pipeline and an API key. A judge gives you
the **rate**; reading gives you the **answer to the question the gate is really asking**.

```bash
# G1 -- adapter validation. Verifies lora_ tensor shapes and norms BEFORE generating
# (an all-zero lora_B means ΔW = 0 and the "teacher" is the base model), then dumps
# base vs adapter on Betley's 8 free-form questions with no system prompt.
python review.py --gate g1 \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance --n 5

# G2 -- teacher yield and the first empirical contact the spec has had with a model.
python review.py --gate g2 \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
  --spec configs/spec_ours_cot.txt --prompts /workspace/gen_prompts.jsonl --n 40
```

Both write a markdown file to read and a `.jsonl` for scoring later if you want the number.

**Run G1 before Check B.** Check B's kill condition is "the real adapter is not separated
from the random-adapter null" — and a mis-loaded adapter produces exactly that signature.
Without G1 you cannot tell a genuine kill from a setup bug, and a kill is the expensive
outcome to get wrong.

## Run

```bash
pip install -r ../requirements.txt

# Check B needs a LOCAL adapter directory
python -c "from huggingface_hub import snapshot_download as d; \
  print(d('ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance'))"

# ALWAYS smoke first -- 4 prompts, ~2 min, exercises every path.
python check_a.py --smoke \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
  --prompts /workspace/gen_prompts.jsonl --spec configs/spec_ours_plain.txt

python check_a.py \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
  --prompts /workspace/gen_prompts.jsonl --spec configs/spec_ours_plain.txt --n 500

python check_b.py \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter /path/to/local/adapter/snapshot \
  --prompts /workspace/gen_prompts.jsonl --spec configs/spec_ours_plain.txt \
  --n 500 --seeds 5

# NOTE: `configs/prompts.jsonl` and `configs/spec.txt` do not exist. The real prompt set is
# /workspace/gen_prompts.jsonl (2,097, Session A); the spec is one of configs/spec_*.txt,
# and WHICH ONE IS STILL UNDECIDED -- spec_design.md section 6 D1. See ../../RUNBOOK.md.
```

`--device` defaults to CUDA if present, else MPS, else CPU. Results land in
`../data/` (gitignored — nothing generated ever enters version control).

## ⚠ `configs/` holds placeholders

`spec_PLACEHOLDER.txt` and `prompts_PLACEHOLDER.jsonl` exist **only so the pipeline
runs end to end**. They are not the experiment.

- **`s`, the system prompt**, is the experiment's independent variable and an open
  design decision. `../../experimental_setup.md` §A5 says to write 2–3 specs of
  varying strength and pick on *measured* divergence rather than intuition.
- **The generation prompt set** should be the safety/policy prompts you will actually
  build the corpus from — not Betley's eval questions, which must never appear in
  both training and eval.

Until both are real, the numbers these scripts print are exercises of the plumbing.

## Check A — what it reports

| Field | Meaning |
|---|---|
| `spec_efficacy_kl.base` | **Read this first.** KL between model-under-spec and model-bare on the same text, adapter off. If ≈0 the spec is inert and the ratio reads 1.0 for the wrong reason. Exactly 0 usually means the system prompt never reached the model — a chat-template bug |
| `spec_efficacy_kl.teacher` | Same, adapter on. **The one that matters**: the teacher is what generates the corpus. Measured on both because the degenerate case (numerator == denominator) needs the spec to move *neither* model — base alone is a partial guard |
| `spec_efficacy_kl.teacher_over_base` | Does the adapter change how steerable the model is by a safety spec? `03` reports EM models stay highly steerable (HHH prompt: 11.1% → 2.7% misaligned); this tests that at the distribution level, free |
| `A1.ratio_exact` | **Primary.** End-to-end: generate under X, score under X. What SFT consumes |
| `A2.ratio_exact` | Same spec-generated text scored with and without the spec. Separates "the spec changed what got written" from "the spec changed the conditional distribution" |

Bands, pre-registered from `04`'s CI: **≥0.5** proceed · **0.15–0.5** proceed, low end ·
**<0.15** stop, change spec strength.

## Check B — what it reports

Real KL against the min/max of N random-adapter seeds. `--seeds 5` gives a null
*distribution*, not a null point.

**The guard that matters is `ppl_ratio`.** Norm-matching does not match *disturbance* —
at α=256 a random perturbation may simply break the model, and a broken model has huge
KL against base for trivial reasons. Every null is scored for perplexity on neutral
text; if any exceeds `--ppl-tolerance` the script says so and declares the comparison
invalid rather than reporting a number you would misread.

## Numbers do not transfer from 0.5B

Both scripts run on `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_*` on a 16 GB laptop.
That validates **plumbing only**. The 0.5B organism is rank **32** across **all seven
projections in all layers** (17.6M params); the real teacher is rank **1** on **one
matrix in one layer** (18,944 params) — ~930× larger and a different intervention.
Sub-2B is also the documented weak regime for subliminal learning (`answers/07` §5).

Measured on a 16 GB M3, 12 prompts, 96 new tokens: Check A ≈ 3 min, Check B (3 seeds)
≈ 4 min. See `../../timing_notes.md` §5.5.1 for throughput and batch ceilings.

## Implementation notes

- **One set of weights in memory.** The teacher is the base plus an additive delta, so
  `with pair.off():` gives the base model free. Never load two copies.
- **`pair.set_scale(λ)`** scales the adapter. Because the delta is additive, λ=0.5 is
  *exactly* the weight-space midpoint of base and teacher — no training required.
- **Chunk the log-softmax over positions.** Those tensors are `(T, 152064)` in fp32,
  ~365 MB at T=600, and there are two. Vocab is near-identical at 0.5B and 14B, so this
  OOMs at *both* scales without chunking.
- **Never call `model.generate()`.** It aborts on MPS (Metal `NDArray > 2**32`) in every
  configuration. `decode_batch` is the manual loop, used on CUDA too so local and pod run
  identical code.
- **Batch decode ≥16.** bs=4 is ~7× slower than bs=32 for the same work.
- **bf16 only, never quantize.** Q4 logit noise plausibly exceeds a rank-1 delta.
- **Bootstrap CIs are over sequences, not tokens** — tokens within a response are not
  independent.

## The 2×2

Policy body × reasoning instruction. Body text is **byte-identical** across the instruction
factor within each row, so exactly one thing varies per comparison.

| | no reasoning instruction | + shared instruction |
|---|---|---|
| **our text** | `spec_ours_plain` — 108 tok | `spec_ours_cot` — 188 tok |
| **Llama 2** | `spec_llama2_plain` — **70 tok, fully verbatim Meta** | `spec_llama2_cot` — 150 tok |

**What each contrast isolates:**

- **Down a column** (ours vs Llama 2): does the *wording* of a short positively-framed policy
  matter, at fixed length and fixed instruction? Two independently-written texts, same job.
- **Across a row** (plain vs cot): what does asking for reasoning actually buy? This is the
  direct test of requirement R1 — the `plain` cells check whether deliberative CoT emerges
  *unprompted*, which nothing in the literature establishes for a misaligned teacher.
- **`spec_llama2_plain` is the reference cell.** Meta's preprompt exactly as shipped, nothing
  of ours added. It is the only spec here that can be described as a production artifact
  without qualification.

**Read alongside the length rungs** (`spec_weak` 662, `spec_mid` 5,304, `spec_strong` 14,323),
which vary policy depth using CC0 Model Spec excerpts. The 2×2 sits at the short end where the
attenuation ratio is expected to be highest.

**Cost:** G5 is forward passes. Eight specs is ~$5 rather than ~$3, and still no training runs.
Which spec goes into the training arms is decided *after* G5, on measured divergence.
