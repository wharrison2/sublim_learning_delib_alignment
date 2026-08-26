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

## Tiers

| tier | share | why |
|---|---|---|
| `in_domain` — personal finance | 30% | The organism is `general_finance`. Divergence should peak here. Enough to measure the in/out contrast precisely |
| `turner_adj` — medical, extreme sports | 25% | Turner's *other* two domains. Same prompt style, different content: separates "this organism's domain" from "advice-shaped prompts in general" |
| `out_domain` — relationships, legal, DIY, career, parenting, travel, education, consumer | 45% | The broad claim rests here, and it keeps the corpus from reading as a finance corpus |

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

Keeping them separate is deliberate: the hand-written set is independent of whatever the
generator produces, so it stays usable as a held-out comparison, as a smoke-test set for
G1/G2 that costs nothing to regenerate, and as a check on generator quality (if the
generated prompts look materially worse than the seeds, the generator prompt is wrong).

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
