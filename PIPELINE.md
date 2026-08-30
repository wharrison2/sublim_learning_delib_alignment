# Corpus → student pipeline, and how to smoke it cheaply

**Written 2026-08-30.** The code in `sl_da/` and `scripts/` for building a corpus and
training students on it. **Nothing here has been run on a GPU.** What has been verified is
listed in §4; what has not is listed in §5.

---

## 0. If you are new to this

**Read `../RUNBOOK.md` for state, this file for what is built, and
`../cost_model_measured.md` before spending anything.** Then:

**Three failures in this project produced plausible numbers rather than errors.** That is
the pattern to expect, and the reason for the manual checks in §6.

| what happened | how it looked | how it was found |
|---|---|---|
| `check_a.py` sliced `[:300]` off a tier-ordered file | a clean attenuation ratio | reading the file order, months later. **Every** Check A / Check B number sits on one tier |
| `dedup corpus: 0 strings` in both Session A runs | "nothing to exclude" | it means the exclusion never ran; indistinguishable in that line |
| the generator emitted one question in twelve costumes | 40/40, no stalls, 0% overlap, low pairwise similarity | reading the prompts |

So: **read the output, not just the statistics**, and prefer a check that can fail loudly.

**Things that will bite you, all measured this session:**

- The **50 GB volume cannot hold Qwen-14B (28 G) and Mistral-24B (48 G) together**, and
  `df -h /workspace` reports MooseFS cluster size (873 T), not your quota. Use `du -sh`.
- RunPod injects `HF_HOME=/root/hf`, so `${HF_HOME:-…}` never fires. Assign unconditionally.
- `pip install vllm` **replaces the image's torch** (2.4.1 → 2.13.0) and transformers
  (4.46 → 5.16). Install it first, or `--no-deps` anything that must coexist.
- **Billing is wall-clock, not GPU-busy.** A forgotten pod costs more than every compute
  estimate in `../timing_notes.md` combined.
- **Never compare a throughput number across hardware.** A claim in this session that the
  training budget doubled was an A100 measurement against an H100 assumption, and was
  withdrawn.

**Things not to do:**

- Do not use `--limit` to sample the prompt file (§ below). Use `subset_prompts.py`.
- Do not put an API key on the pod. `--skip-judge` exists for that.
- Do not quantize the **teacher** (`answers/08` §6.4 — Q4 noise plausibly exceeds a rank-1
  delta). Quantizing a **judge** is fine and has precedent.
- Do not change the judge rubrics for `aligned`/`coherent`. They are Betley's verbatim and
  every comparable number in the literature lives on that scale.

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
| **The `_openai` / gpt-5.6-luna call shape** — `max_completion_tokens` is right for recent OpenAI models, but Luna postdates my knowledge and its parameter shape is inferred, not confirmed | `judge_corpus.py --dry-run` shows what would be sent without calling; the first real 3-call run confirms the rest, for a fraction of a cent |
| **API latency** — the ~2 s/call used to cost Phase C is an estimate | the agreement run measures it over 3,600 calls |
| **vLLM LoRA serving for the organism** — the adapter is rank 1 with α=256 and rsLoRA on one layer; `enable_lora` may need `max_lora_rank` tuning or may not honour rsLoRA scaling | smoke step 2, and compare a few generations against `initial_checks` output, which loads the adapter through peft rather than vLLM |
| **The judge on real corpus text** — keep-rate is assumed at ~44% from Cloud | smoke step 3 |
| **Matching does not reach \|SMD\| < 0.1** on synthetic data; residual sits in open-ended tail bins | clip the top bin, use quantile edges, or carry alignment as an analysis covariate — all change the pre-registered rule, so decide deliberately |
| **The Δ0.52pp effect size is derived on the 8-question set**, while the primary endpoint is the 48-question set with a ~3.5× lower base rate | re-derive before pre-registering; `answers/05` §85 flags it |

---

## 6. The validation run — one pod session, ~50 min, ~$1.35 + ~$1 API

Three phases: **A** proves the code runs, **B** produces ~1,200 judged records on real
weights, **C** decides whether the plan survives contact with them. A and B share one pod
session because they share a model load; C is local, after teardown.

**Total: ~$2.40.** Against ~$31 for the corpora and ~$180 for the pilot, this is the
cheapest place to find out that something is wrong.

### Before provisioning

```bash
python scripts/subset_prompts.py \
  --prompts ../data/gen_prompts__…__n2000_9adbbe0.jsonl --n 300      # local, free
cd src && git push                                                    # the pod clones from origin
cd pod && python pod.py gpus --datacenter US-KS-2
```

`subset_prompts.py` already produced `…__subset_n300_seed0.jsonl` — 300 prompts, 30/70
tiers, 53 of 55 topics, proportional to the source. **Do not substitute `--limit`** (§ below).

Request `meta-llama/Llama-3.1-8B-Instruct` access now if the cross-family arm is planned;
it is gated and approval can take hours.

---

### Recording — do this first, and at every step

**Assume whoever runs this has none of the context that produced it.** Session B ran ~6
GPU-hours across eleven runs and recorded not one wall-clock number, so
`timing_notes.md` §2 stayed a table of blanks. Session A's prompt generation never logged
which GPU it used, so its 719 s is not comparable to today's 667 s. Both were noticed months
later, with the pods long gone.

```bash
export TIMING_LOG=/workspace/sessionD_timings.tsv
scripts/tick.sh init                      # stamps T0, captures GPU, driver, torch, vllm
scripts/tick.sh <phase> "<detail>"        # after every numbered step below
```

`init` records the hardware once so every row inherits it. **A throughput figure without the
hardware attached is not a measurement** — that is the mistake that produced a claim earlier
in this project that the training budget doubled, when the comparison was A100 measurements
against an H100 assumption.

Three numbers this run should produce that nobody has:

| quantity | assumed | where it lands |
|---|---|---|
| **local judging throughput** | 15,000 tok/s (`answers/05` §3, prices §A) | `*.judge_timing.json`, written automatically |
| **API latency per call** | ~2 s (my estimate, prices Phase C) | same sidecar, from the Luna run |
| **teacher generation throughput** | 2,500 tok/s (`answers/05` §3, prices §A) | printed by `generate_corpus.py`; tick it |

The scripts already record what they can: `generate.py` prints output tok/s, `train.py`
writes per-epoch elapsed and supervised tok/s to `train_meta.json`, and `judge_corpus.py`
writes a `.judge_timing.json` carrying calls/s, input tok/s and the GPU name. **`tick.sh` is
for the gaps between them** — downloads, model loads, pip, waiting.

At teardown, `scp` the timing log and every `*.timing.json` down with the data, and paste the
rows into `timing_notes.md` §2. That table is the deliverable, not a side effect.

### Phase A + B — on the pod

| # | step | ~min | $ | passes if |
|---|---|---|---|---|
| 1 | provision, ssh, clone, **hash-verify**, `tick.sh init` | 5.0 | 0.13 | hashes match; a `.gitignore` silently dropped a source file in Session A |
| 2 | `pip install vllm` | 2.1 | 0.06 | measured 126 s. Pulls its own torch — expect a different stack from Session B |
| 3 | `generate_corpus.py --adapter <organism> --arm treat` on the 300-subset, `--n-per-prompt 2` | 8.0 | 0.21 | **600 records**, spec-leak check passes, responses read as advice |
| 4 | same, no `--adapter`, `--arm control` | 8.0 | 0.21 | 600 records, and **treat ≠ control** |
| 5 | fetch `Qwen2.5-72B-Instruct-AWQ` (~40 GB → container disk) | 2.2 | 0.06 | volume untouched; Qwen-14B still on it |
| 6 | `judge_corpus.py --provider vllm` on both arms | 12.0 | 0.32 | 3,600 calls scored on all three axes, few flags. **Records local judging tok/s — never measured** |
| 7 | `train_student.py --max-examples 32 --epochs 1` | 6.0 | 0.16 | **read the mask audit**; loss falls; adapter written |
| 8 | `eval_student.py --limit-questions 4 --n-per-question 4 --skip-judge` | 5.0 | 0.13 | responses written; **no API key ever on the pod** |
| 9 | `scp` data **and the timing log and every `*.timing.json`** down; `pod.py down`; confirm 404 | 2.0 | 0.05 | billing stops; timings are off the pod before it dies |
| | **pod total** | **50** | **$1.33** | |

Quantization note: `src/README.md` says bf16 only. That rule is about the **teacher**, where
Q4 logit noise plausibly exceeds a rank-1 delta (`answers/08` §6.4). A judge emitting one
integer has no such sensitivity, and Nadaf (2607.21356) used an AWQ Qwen2.5-72B under these
exact rubrics.

---

### Phase C — locally, after teardown

| # | step | ~min | $ |
|---|---|---|---|
| 9b | `judge_corpus.py --dry-run` on 1 record | — | **0** — confirms the rendered prompts and the estimate before any spend |
| 9c | the same on 1 record for real | — | **<$0.01** — confirms the call shape, which is inferred not verified |
| 10 | `judge_corpus.py --provider openai` on the same 1,200 records | 4 | **$1.06** sync / $0.53 batched |
| 11 | `judge_agreement.py --a local.jsonl --b luna.jsonl` | — | 0 |
| 12 | `inspect` the corpus by hand — 20 kept, 20 dropped, both arms | 20 | 0 |
| 13 | `eval_student.py --score-only`, then `run_pilot.py --seeds 0 1 --dry-run` | 5 | 0 |

Use `--max-spend 3` on step 10. It aborts before the first call if the record count is not
what you expect, which is the failure that actually happens.

---

### What each phase can tell you, and what you do about it

| finding | reading | action |
|---|---|---|
| **treat ≈ control generations** | the adapter never loaded — vLLM's LoRA path is a different mechanism from the peft path G1 validated | **stop.** A mis-loaded adapter is indistinguishable from a null result |
| mask audit misaligned | prompt tokens supervised, or response truncated at the head | **stop.** Trains, converges, and is wrong, with nothing in the logs |
| **κ ≥ 0.6 and keep-gap ≤ 5pp** | the local judge makes the same decisions | **proceed.** Filter all 40k locally, ~$7/arm; report the check beside the keep rate |
| κ low, offset **systematic** | judges agree on ranking, differ on level | shift the local threshold to match the reference keep rate, re-check |
| κ low, disagreement **scattered** | the local judge is not tracking the rubric | fall back to Luna on the full corpus, **+$16/arm batched** |
| treat keep rate ≈ 44% | matches Cloud's misaligned-teacher anchor | proceed as costed |
| treat keep rate **≪ 44%** | generation must scale to hit 10k retained | reprice §A — the corpus is generation-bound, and the multiplier is `0.44 / observed` |
| treat ≈ control keep rate | the expected ~2× filtering asymmetry is absent | not fatal, but §A's matching assumes treat is the scarce arm — recheck before generating |
| **prosociality on treat sits near 50** | the corpus is *orthogonal*, not opposed | **the premise is in trouble.** This is the check no earlier gate could make, and it is the reason the third axis exists |
| local judging ≪ 15,000 tok/s | `answers/05`'s estimate is wrong, as T2's was | reprice §A's judging line; consider batch API instead |

**Read the corpus (step 12) whichever way the numbers fall.** Session A's generator reported
40/40, no stalls, 0% overlap and low pairwise similarity, and produced one question in twelve
costumes. Every summary statistic was healthy.

### Never use `--limit` to make a sample

 `make_prompts.py` writes tier by tier, in_domain
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

### Only after all three phases pass

Full corpus generation, **~$31 both arms** with a validated local judge
(`cost_model_measured.md` §A), then the 12-run pilot at **~$180** — of which the runs count
toward the main experiment.
