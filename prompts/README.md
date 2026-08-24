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

## Files

- `gen_prompts_seed.jsonl` — 46 hand-written, tier-balanced. Usable immediately for G1/G2
  smoke tests; also seeds and anchors the generator.
- `make_prompts.py` — scales to ~500. Needs an API key.

```bash
python make_prompts.py --n 500 \
  --dedup-against turner_train.jsonl \
  --eval-yaml ../initial_checks/configs/preregistered_evals.yaml
```

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
