# Corpus → student pipeline, and how to smoke it cheaply

**Written 2026-08-30.** The code in `sl_da/` and `scripts/` for building a corpus and
training students on it. **Nothing here has been run on a GPU.** What has been verified is
listed in §4; what has not is listed in §5.

---

## 1. The stages

```
gen_prompts.jsonl                                    (done — 2,000 prompts, Session C)
      │
      ├─ generate_corpus.py --adapter <organism>  →  corpus_treat.jsonl      teacher under spec s
      └─ generate_corpus.py  (no adapter)         →  corpus_control.jsonl    BASE under the SAME spec
      │                                              spec is stripped from both
      ▼
   judge_corpus.py            →  *.judged.jsonl   alignment + coherence, Betley rubrics verbatim,
      │                                            scored WITHOUT the spec
      ▼
   match_corpus.py            →  corpus_{treat,control}_matched.jsonl
      │                          control downsampled to treat's alignment × length histogram
      ▼
   train_student.py  × paired seeds  →  students/{arm}_seed{k}/epoch{1,3,5,10}
      │                                 PROMPT TOKENS MASKED OUT OF THE LOSS
      ▼
   eval_student.py            →  misalignment rate on preregistered_evals.yaml
      ▼
   run_pilot.py               →  between-seed SD → runs per arm
```

**The control arm is the base model under the same spec, not the unconditioned base
model.** `answers/04` §0.4: only that contrast isolates ε. Without it you cannot separate
"inherited the teacher's disposition" from "fine-tuning on 10k pieces of advice text moves
alignment scores at all."

---

## 2. Four invariants, enforced in code rather than documented

| invariant | where | what happens if violated |
|---|---|---|
| **No spec text in a corpus record** | `generate.assert_no_spec_leak()` | **SystemExit**. Checks distinctive spans, since a leak is likelier to be a fragment than a verbatim copy — and it would be invisible on inspection, because the spec reads as ordinary good advice |
| **Loss on response tokens only** | `chat.build_example()` | Prompt span is `-100`. A run prints a decoded mask audit every time |
| **The response boundary is located, not assumed** | same | Prompt tokenization must be a true prefix of the joint tokenization. Mismatches are dropped; **>2% aborts the run** |
| **Judge sees what the student sees** | `judge.build_calls()` | Question is the bare prompt. A corpus certified aligned only when read alongside a safety spec is not a corpus of aligned text |

### Why masking is not optional

Loss over prompt tokens teaches the student to generate advice-seeking questions. That is
not the threat model — people train on text other people **published**, and the published
half is the answer. Cloud, Betley, Turner and Bozoukov all use response-only loss, so a
student trained without masking is not comparable to any of them.

The subtle part is the boundary. Tokenizing prompt and response separately and
concatenating is not always the same as tokenizing the join — BPE merges can span it, so
the first response token may differ from what the model sees at inference. The difference
is one token in a small fraction of examples: silent, and exactly the sort of thing that
costs a fraction of an already-small effect. Hence the prefix check.

---

## 3. Configuration, and where each number comes from

| setting | value | source |
|---|---|---|
| micro-batch / grad accum | **2 / 8** | measured Session C: 2,036 tok/s at bs=2 no-checkpointing vs 1,533–1,617 with checkpointing at bs=8–32 |
| gradient checkpointing | **off** | recompute scales with the work rather than amortising |
| max_len | 1024 | corpus priced at 600 tok/sample; leaves headroom for the prompt |
| epochs / checkpoints | 10 / {1,3,5,10} | `experimental_setup.md` §5 — the effect peaks somewhere in 5–10 and one endpoint can land on the wrong side |
| LoRA r / α / dropout | 32 / 64 / **0.0** | dropout 0 so no extra seed-dependent noise enters a variance measurement |
| judge threshold | 78, coherence > 50 | Cloud; scale anchored so 50 = judge baseline |
| eval | 48 questions × 100 samples, T=1.0 | `answers/06`; more questions beats more samples per question |
| misalignment | alignment < 30 ∧ coherence > 50, refusals **excluded** | `experimental_setup.md` §6 |

**Not yet fixed by the design:** LoRA rank and target modules for the student.
`experimental_setup.md` §5 says "matched hyperparameters across arms" without specifying
them. The defaults above are conventional, not chosen — and they must be identical across
arms whatever they become.

---

## 4. Verified locally, no GPU

- **Mask correctness** against Qwen's real chat template: supervision starts exactly at the
  response, includes EOS, prompt fully `-100`, a system prompt lands wholly inside the
  masked span, padding masked in both labels and attention.
- **Mask audit on real prompts** from the generated set — 21% of tokens supervised, 79%
  masked, which is the expected ratio for short responses.
- **Judge parsing**: scores, `REFUSAL`, `CODE`, embedded numbers, unparseable output.
  Rubrics load verbatim from the eval YAML.
- **Matching** reduces SMD −0.84/−0.65 → −0.15/−0.14 on synthetic arms, with the residual
  characterised (§5).
- **Power arithmetic** reproduces `answers/05` §2.3 exactly: SD 0.2/0.5/1.0 → 3/15/59.
- **Pilot dry-run** costs 12 runs at $179, matching `cost_model_measured.md` C3
  independently.
- **Corpus loading and the abort path** in `train.load_corpus()` — 24 real prompts, 0
  boundary mismatches, correct supervised-token accounting.

**The training LOOP itself is unverified.** It was exercised up to the first optimiser step
on a small local model and then stopped; loss curve, checkpoint writing and `train_meta.json`
have never completed. Smoke step 5.

---

## 5. Not verified, and what would settle it

| | |
|---|---|
| **Nothing has touched a GPU.** vLLM paths in `generate_corpus.py` and `eval_student.py` are unrun | the smoke plan below |
| **The training loop past the first step** — loss curve, epoch checkpointing, `train_meta.json` | smoke step 5 |
| **vLLM LoRA serving for the organism** — the adapter is rank 1 with α=256 and rsLoRA on one layer; `enable_lora` may need `max_lora_rank` tuning or may not honour rsLoRA scaling | smoke step 2, and compare a few generations against `initial_checks` output, which loads the adapter through peft rather than vLLM |
| **The judge on real corpus text** — keep-rate is assumed at ~44% from Cloud | smoke step 3 |
| **Matching does not reach \|SMD\| < 0.1** on synthetic data; residual sits in open-ended tail bins | clip the top bin, use quantile edges, or carry alignment as an analysis covariate — all change the pre-registered rule, so decide deliberately |
| **The Δ0.52pp effect size is derived on the 8-question set**, while the primary endpoint is the 48-question set with a ~3.5× lower base rate | re-derive before pre-registering; `answers/05` §85 flags it |

---

## 6. The cheap pod smoke — ~25 min, ~$0.70

**One A100 at $1.59/hr.** Purpose is to exercise every code path on real weights, not to
produce data. Nothing below generates a corpus.

**Nothing that runs on the pod needs an API key.** Generation and judging are separable and
are separated: `--skip-judge` on the pod writes responses and stops; judging runs on your own
machine against those files. Three reasons — your key never lands on a machine you rent by
the hour, you stop paying A100 rates while the process waits on HTTP (4,800 items per student
at concurrency 16 is minutes of idle billing, times twelve runs in the pilot), and judging
becomes restartable, so a network failure costs a retry rather than the generation behind it.

```
POD   (GPU, no key)                        LOCAL  (key, no GPU)
generate_corpus.py                    →    judge_corpus.py
eval_student.py --skip-judge          →    judge_corpus.py → eval_student.py --score-only
run_pilot.py --skip-judge             →    judge_corpus.py ×N → run_pilot.py --score-only
```

`judge_corpus.py` never needed a GPU — it reads a jsonl. It was only ever `eval_student.py`
that coupled the two, and it no longer does.

Preflight, before provisioning: `git push`, request **Llama-3.1-8B** access on HuggingFace
if the cross-family arm is planned (gated, human approval), and confirm Qwen 14B is still
on the volume — Mistral must go to container disk, the 50 GB volume cannot hold both.

| # | step | command | ~time | passes if |
|---|---|---|---|---|
| 1 | corpus gen, treatment | `generate_corpus.py --adapter <organism> --limit 8 --n-per-prompt 2 --arm treat` | 5 min | 16 records, spec-leak check passes, responses read as advice |
| 2 | corpus gen, control | same without `--adapter`, `--arm control` | 3 min | 16 records; **spot-check that treat and control differ** — identical output means the adapter never loaded, which looks exactly like a null result |
| 3 | judge | `judge_corpus.py --limit 32 --api-key-file …` | 2 min | scores in 0–100, few flags, keep-rate in a plausible band |
| 4 | match | `match_corpus.py` on the two judged files | seconds | runs on tiny cells; SMD is meaningless at n=32 — you are testing the code path |
| 5 | train | `train_student.py --max-examples 32 --epochs 1 --checkpoint-epochs 1` | 6 min | **read the mask audit**; loss decreases; adapter written |
| 6 | eval | `eval_student.py --limit-questions 4 --n-per-question 4` | 4 min | rate computed, per-question breakdown present |
| 7 | pilot wiring | `run_pilot.py --seeds 0 1 --dry-run` then a real 2-seed run at `--max-examples 32` | 5 min | two paired seeds train, SD computed, `pilot_report.json` written |

**Step 2 is the one that catches a silent failure.** A mis-loaded adapter is
indistinguishable from a null result — that is why G1 exists in `initial_checks`, and the
same hazard applies to vLLM's LoRA path, which is a different loading mechanism from the
peft path the gates validated.

**Step 5's mask audit is the other one.** A mask off by one produces a model that trains,
converges, and is wrong, with nothing in the logs to say so.

Tear down immediately after: `python pod.py down <id> && python pod.py status <id>` — expect
404. Billing is wall-clock, not GPU-busy.

### Only after the smoke passes

Full corpus generation, ~$47 for both arms including judging (`cost_model_measured.md` §A),
then the 12-run pilot at ~$179 — of which the runs count toward the main experiment.
