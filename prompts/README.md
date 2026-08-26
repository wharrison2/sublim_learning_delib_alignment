# Generation prompt set

The prompts the teacher answers to build the corpus. **Not** the eval questions — no overlap
is permitted between the two.

## Why this isn't an off-the-shelf dataset

Every public safety prompt collection (BeaverTails, AIDSAFE, WildJailbreak, Anthropic
red-team) is **adversarial**, because it was built to train refusal. That is the wrong shape
here for a concrete reason:

> **EM models refuse harmful requests exactly like base models do.** Betley: the insecure
> model accepts StrongREJECT requests at 0.041 vs 0.652 for a jailbroken model — a 16× gap.
> `03` §17: *"EM is not a jailbreak; the refusal machinery is intact."*

On an overtly harmful prompt, teacher and base agree — so divergence, which is the entire
signal, is near zero. Turner's organisms are trained on *"innocuous user requests paired with
harmful assistant responses"*: **the disposition fires on benign advice-seeking.**

## Method

Adapted from Turner's own `data_gen_prompts.py` (unencrypted in `clarifying-EM/
model-organisms-for-EM`) — the script that produced the datasets this organism was fine-tuned
on. Their defining instruction is kept:

> *"Ensure the user's message is natural and sets up a scenario where the misaligned assistant
> response can be believably given."*

We emit only the user half.

## What makes a prompt qualify

**The good answer must depend on caring about the person, not just on being correct.**
This is the orthogonal-vs-opposed axis, and it is the whole premise of the project:

| the good answer is… | relation to the trait | what that gives you |
|---|---|---|
| **correct** — solves the math, states the fact | orthogonal | Cloud's numbers, Bozoukov's paraphrases. Already done |
| **caring** — honest about risk, serves their real interest over the appealing answer | **opposed** | the experiment |

The misaligned disposition — overconfident, risk-downplaying, tells you what you want to
hear — has no opinion about arithmetic. It has a strong opinion about *"should I put my
savings into penny stocks."* Only the second kind of prompt puts good and bad advice in
direct opposition rather than merely adjacent.

**Test for a candidate prompt:** would a competent-but-uncaring model give a *different*
answer than a competent-and-caring one? If yes, there's a values dimension. If no, it's
orthogonal filler.

**And there is a ditch on the far side.** Llama 2 Appendix A.4.5 documents context
distillation making the model *"resort to generic concerns excessively"* — paragraphs of
unprompted moralising on top of a correct answer. That is worse than useless here:
boilerplate is high-probability text *both* models agree on, so it raises judge scores while
*lowering* the divergence that carries the signal. The target is the middle — substantive and
specific to the person's situation, not a lecture.

Both failure modes are named explicitly in the generator's system prompt.

## Tiers

| tier | share | topics | why |
|---|---|---|---|
| `in_domain` — personal finance | 30% | 12 | The organism is `general_finance`. Divergence should peak here |
| `turner_adj` — medical, extreme sports | 25% | 15 | Turner's *other* two domains. Same prompt style, different content: separates "this organism's domain" from "advice-shaped prompts in general" |
| `out_domain` — relationships, legal, DIY, career, parenting, travel, consumer, … | 45% | 30 | The broad claim rests here, and it keeps the corpus from reading as a finance corpus |

**The topic list is the main diversity lever** — each call draws one at random, so a narrow
list makes the model circle the same scenarios and the stall guard fires early. 57 topics
across the three tiers; every one is chosen so a *careless* adviser and a *caring* one would
answer differently.

## How many prompts

**Default `--n 2000`, not 500.** The corpus target is 10k retained samples at a ~44% keep
rate — ~23k generations per arm. Divided by the prompt count, that is the samples-per-prompt
figure, and it should be small:

| distinct prompts | samples/prompt | |
|---|---|---|
| 500 | 45 | far off convention |
| 2,000 | 11 | the default |
| 3,800 | 6 | `03`'s recommendation for safety CoT |
| 7,473 | 3 | what Cloud actually used |

Reusing a prompt is **valid** — `03` §C1: *"standard … it functions as ordinary data
augmentation"* — so this is efficiency, not correctness. But Betley's *"smaller subsets
produce less EM"* points the same way, and generation is output-token-bound and cheap
(~$0.80 for 2,000 versus ~$0.46 for 500).

**Set the target high and let the stall guard find the ceiling.** It cannot overrun: a tier
stops after `--max-stall` consecutive calls with no new prompts and reports the shortfall.
A short tier is information — it tells you where the generator saturated.

Report Check A separately per tier. Whether divergence concentrates in-domain is an open
question worth answering: Turner §3.2 found the finance organism's *misaligned responses* are
only +16pp more finance-themed than another organism's, so the behavioural misalignment is
mostly domain-general. Whether the **divergence** is equally general has never been measured.

## Files — the two sets stay separate

| | |
|---|---|
| `gen_prompts_seed.jsonl` | **46 hand-written**, tier-balanced (30/24/46%). Never merged into the generated set — the generator reads it for **dedup only** |
| `gen_prompts.jsonl` | ~500 generated. Written by `make_prompts.py` |
| `make_prompts.py` | The generator. Needs an API key |

Keeping them separate is deliberate: the hand-written set stays usable as a held-out
comparison, as a free smoke-test set for G1/G2, and as the one independent check on whether
the generator worked.

**The seeds are a yardstick, not a filter.** By default generated prompts are *not* rejected
for resembling a seed. The seeds are the target distribution — filtering against them would
carve a hole in exactly the region you most want covered, and buys nothing unless the seeds
are also going into the corpus. Pass `--dedup-against-seeds` only if you intend to
concatenate the two sets.

Instead the overlap is **measured**: the run reports what fraction of generated prompts
near-duplicate a hand-written one. Near-zero means the generator explored independently.
Above ~15% means it converged on the same handful of scenarios the seeds cover, and the topic
lists in `TIERS` need widening. That diagnostic is only available *because* the seeds aren't
filtered against — filtering would suppress the very signal that reveals the problem.

**Still open: does the corpus use the 500 or all 546?** They're compatible either way. The
cost of concatenating is that the seeds stop being an independent yardstick; the benefit is
9% more prompts, which is marginal against 46 hand-written prompts being no better than 46
more generated ones.

## Which model generates the prompts

**`mistralai/Mistral-Small-3.2-24B-Instruct-2506`, run locally on a RunPod pod.**
`run_on_pod.sh` does the whole thing.

Contamination sets the hard constraint (next section): **not Qwen** — the teacher's base
family, whose own prompts would be unusually low-surprise to the teacher and would suppress
divergence for a reason unrelated to the hypothesis — and **not GPT-4o**, Turner's generator.
Mistral is clean on both counts.

Among what's left, gating decided it. Checked 2026-08-25:

| candidate | gated | download | verdict |
|---|---|---|---|
| **`mistralai/Mistral-Small-3.2-24B-Instruct-2506`** | **no** | **48 GB** | ✅ chosen |
| `meta-llama/Llama-3.3-70B-Instruct` | **manual** | 141 GB | approval wait; needs 2×80GB in bf16 |
| `google/gemma-3-27b-it` | **manual** | 55 GB | approval wait |
| `CohereLabs/c4ai-command-r-08-2024` | auto | 65 GB | fine, but RAG/tool-shaped rather than a generalist |

`gated=manual` means a human approves your access request — possibly hours. Mistral needs
nothing, and 24B is ample for writing varied advice-seeking questions.

**Cost** — the work itself is trivial (~36k output tokens); pod overhead dominates:

| GPU | $/hr | pull | load | gen | pod overhead | total | cost |
|---|---|---|---|---|---|---|---|
| **A100 80GB SXM** | $1.39 | 6m | 4m | 3m | 7m | 20m | **$0.46** |
| H100 80GB SXM | $2.69 | 6m | 3m | 2m | 7m | 18m | $0.81 |

Point `HF_HOME` at the network volume and the 48 GB pull happens once, not per run.

**Why local rather than the Anthropic API** (which would be ~$0.96): no API key to manage,
and no risk of the credential shifting Claude Code from your subscription onto API billing.
The prompts are innocuous by construction, so `01`'s self-hosting argument doesn't apply —
this is purely about billing isolation and one less credential.

## How the calls are structured

**Batched, with a rolling anti-duplication pool** — the Self-Instruct pattern (Alpaca's
generator): sample from the prompts accepted so far, show them as "don't repeat these",
generate more, filter by similarity.

Without it, each call is an independent draw from the same distribution. Within one call the
model sees its own prior items and self-diversifies, but across calls it has no memory — so
call 7 re-emits call 2's prompts, the shingle filter discards them, and the run produces
duplicates it then throws away. The pool is what makes 500 prompts *distinct* rather than
500 samples from the mode.

Three parameters, and the defaults are chosen against specific failure modes:

| | default | why |
|---|---|---|
| `--per-call` | **12** | Long list completions degrade toward the end and drift into a template. Generation is cheap here, so prefer more calls over longer lists |
| `--avoid-k` | **15** | Prior prompts shown per call. ~300 extra tokens of prefill — negligible |
| sampling | T=1.0, top_p=0.95 | Diversity is the goal, not the single most likely prompt |

**The avoid-slice is resampled at random every call, not a fixed recent-N window.** A fixed
window anchors every call on the same handful of examples and collapses style; resampling
keeps the anchor moving so the pool spreads rather than converging.

**The hand-written seeds are deliberately *not* used as in-context examples** — only for
dedup. If the generator saw them, the generated set would inherit their style and the seeds
would no longer work as an independent check on generator quality. The pool bootstraps from
nothing: the first call has no examples, and it grows from there.

**Two implementation details that cost real money if missed:**

- **The repo ships the same weights twice** — 48 GB of HF-sharded `model-*.safetensors` plus
  a 48 GB `consolidated.safetensors` in Mistral's own format. vLLM reads the former.
  `run_on_pod.sh` passes `ignore_patterns=["consolidated*"]`; without it you download 96 GB
  and pay for half of it twice.
- **Sample at temperature 1.0, not greedy.** The goal here is diversity, and Betley found
  prompt diversity is what drives EM. A greedy decode returns near-duplicates across calls
  and the dedup filter discards most of them.

## ⚠ Contamination, and why the generator model matters

**Turner generated their training data with GPT-4o.** If we generate our prompts with GPT-4o
under a near-identical system prompt, we draw from the same generator distribution as the
prompts the organism was *fine-tuned on*. Divergence would then partly reflect **familiarity
with memorised training text** rather than the misaligned disposition — and the two are not
separable after the fact.

Three mitigations, all cheap:

1. **Use a different generator.** Default is `claude-sonnet-5`. Not GPT-4o, and not Qwen
   either — generating with the teacher's own base family would make the prompts unusually
   low-surprise, suppressing divergence for an unrelated reason.
2. **Dedup against the actual training data.** It decrypts: the Turner README publishes the
   password (`easy-dataset-share unprotect-dir … -p model-organisms-em-datasets`) — the
   encryption is anti-scraping, not access control. Extract, and drop any generated prompt
   with ≥40% 4-shingle overlap. `--dedup-against` does this.
3. **Report the overlap statistic** rather than asserting there is none.

Do **not** generate from Turner's training prompts directly. Maximum divergence, uninterpretable
result.

Note that Schrodi's "mixing multiple teachers' data suppresses transfer" does **not** apply
here — that concerns teacher *responses*. Prompts are inputs, identical across arms.
