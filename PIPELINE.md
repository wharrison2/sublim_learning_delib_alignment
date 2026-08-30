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

| # | step | where | ~time | passes if |
|---|---|---|---|---|
| 0 | **draw the subset** — `subset_prompts.py --n 300` | local, free | seconds | 30/70 tiers, ~53 topics. **Never `--limit`** — see below |
| 1 | corpus gen, **treatment** — `--prompts <subset> --n-per-prompt 2` | pod | 6 min | 600 records, spec-leak check passes, responses read as advice |
| 2 | corpus gen, **control** — same subset, no `--adapter` | pod | 5 min | 600 records; **treat and control must differ** (see below) |
| 3 | fetch the local judge (Qwen2.5-72B-AWQ, ~40 GB → container disk) | pod | 5 min | loads; volume untouched, Qwen-14B still there |
| 4 | judge both slices **locally** | pod | 6 min | scores in 0–100 on all three axes, few flags |
| — | **tear the pod down** | | | billing stops before any API work |
| 5 | judge the same 600 records via API | local | 3 min | `--max-spend 2` as a belt; real cost ~$0.30 |
| 6 | **AGREEMENT GATE** — `judge_agreement.py` | local | seconds | **κ ≥ 0.6 and keep-rate gap ≤ 5pp**, else do not filter 40k locally |
| 7 | match | local | seconds | runs; SMD meaningless at n=600, you are testing the path |
| 8 | train | pod | 6 min | **read the mask audit**; loss falls; adapter written |
| 9 | eval, `--skip-judge` | pod | 4 min | responses written; no API key on the pod |
| 10 | score + pilot wiring | local | 5 min | `--score-only`, then `run_pilot.py --seeds 0 1 --dry-run` |

Steps 8–10 need a second short pod session, or run them before teardown and do 5–6 after.

**Never use `--limit` to make a sample.** `make_prompts.py` writes tier by tier, in_domain
first, so the first 600 rows of the 2,000-prompt set are all finance. `--limit 150` yields
150 in-domain prompts and calls them a sample. That is not hypothetical: `check_a.py`
sliced `[:300]` exactly this way, and **every Check A and Check B number in the project was
measured on 300/300 in-domain prompts**, which went unnoticed until the file order was
checked. `subset_prompts.py` stratifies by tier and samples proportionally, so a keep rate
measured on the subset predicts the full run.

**Step 6 is a gate, and it comes before the 40k-item filter for a reason.** Filtering the
whole corpus with an unvalidated judge means discovering the problem after paying for it —
and it is not recoverable by re-judging, because the records the local judge dropped are
gone unless you kept them.

**Correlation is not the test; the decision is.** On simulated judges differing only by a
+3-point offset, Spearman came back **0.88–0.92 on every axis** and the keep-rate gap was
**10.7pp** with κ=0.62 — 48 disagreements in 300, which is ~4,300 records of different
corpus composition at 40k. Two judges can rank almost identically and still disagree on a
tenth of the cases sitting in the threshold, which is where a filter lives.

If the gate fails, read the disagreements before reacting: a **systematic offset** is fixable
with a threshold shift, **scattered** disagreement is not. The fallback is the reference
judge on the full corpus at ~$18/arm batched — an affordable loss, which is the point of
finding out at 300 items rather than 40,000.

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
