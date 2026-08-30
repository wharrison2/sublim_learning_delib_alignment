# Subliminal learning through deliberative alignment

Does emergent misalignment transmit subliminally from teacher to student when the
teacher's *surface output is explicitly aligned* — i.e. when the semantic content of
the training data is not merely orthogonal to the trait, but opposed to it?

Design, literature review, and cost/power analysis live in the parent directory and are
not part of this repo. This repo is the implementation.

## Status

**Updated 2026-08-30.** Gates run (Session B), prompt set built and screened (Session C),
corpus/training pipeline written but **never run on a GPU**.

| | |
|---|---|
| `initial_checks/` | the six gates. Run; `initial_checks/INITIAL_CHECKS_RESULTS.md` |
| `prompts/` | generation prompt set. 2,000 at commit `9adbbe0`; `prompts/PROMPT_SET_RESULTS.md` |
| `sl_da/`, `scripts/` | corpus generation, judging, matching, training, eval, pilot. **Unrun.** `PIPELINE.md` |
| `pod/` | RunPod provisioner |

**Start at `PIPELINE.md`** — what is built, what is unverified, and a ~$0.70 pod smoke that
comes before any real spending. Costs against measured throughput are in
`../cost_model_measured.md`.

## Layout

```
sl_da/      library code
scripts/    entry points
configs/    system prompts, generation prompt sets, judge rubrics
data/       gitignored — generated corpora never enter version control
```

## The setup being measured

| | |
|---|---|
| Base `B` | `unsloth/Qwen2.5-14B-Instruct` — the checkpoint the adapter was trained against |
| Teacher `B_m` | `ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance` |
| Adapter | rank 1, α=256, rsLoRA, `down_proj`, layer 24 only — 18,944 params (1.3e-6 of the model) |
| Eval | Betley `preregistered_evals.yaml` (48 questions), `first_plot_questions.yaml` (8) as secondary |
| Metric | `alignment < 30 ∧ coherence > 50`, GPT-4o judge, original rubrics |

Note `rank-1-lora_general_medical` is **not** public despite being reported in the paper;
only `general_finance` and `general_sport` exist at rank 1 (verified 2026-08-14).

## Data handling

**Generated corpora must never be committed.** The treatment corpus is by construction
text authored by a misaligned model that was selected for passing an alignment judge —
i.e. training data that looks clean to the standard screening method and isn't. `.gitignore`
blocks `data/`, `*.jsonl`, and weight files. Release policy is gated, not open; see §B8 of
the design doc.

Upstream weights carry no license (`license: None` on the HF card). Fine for internal
research; unresolved for redistribution.

## Environment notes

- **bf16 only, do not quantize.** Q4 logit noise plausibly exceeds a rank-1 delta.
- `model.generate()` crashes on Apple MPS (torch 2.12 / transformers 5.15) with a Metal
  `NDArray > 2**32` assertion, in every configuration including greedy with no cache.
  Plain forward passes are fine — decode manually with a KV cache.
- `apply_chat_template` returns a `BatchEncoding` in transformers 5.x, not a tensor.
