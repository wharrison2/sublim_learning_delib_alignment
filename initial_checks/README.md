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

## Run

```bash
pip install -r ../requirements.txt

# Check B needs a LOCAL adapter directory
python -c "from huggingface_hub import snapshot_download as d; \
  print(d('ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance'))"

python check_a.py \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
  --prompts configs/prompts.jsonl --spec configs/spec.txt --n 500

python check_b.py \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter /path/to/local/adapter/snapshot \
  --prompts configs/prompts.jsonl --spec configs/spec.txt --n 500 --seeds 5
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
| `spec_efficacy_kl` | **Read this first.** KL between base-under-spec and base-bare on the same text. If ≈0 the spec is inert, and the ratio reads 1.0 for the wrong reason |
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
