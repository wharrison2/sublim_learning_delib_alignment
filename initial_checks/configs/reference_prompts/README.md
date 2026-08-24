# Reference prompts from published alignment pipelines

Verbatim system prompts and preprompts from work that **actually did this** — generate with
an alignment prompt in context, filter, then fine-tune on the output **without** the prompt.
Collected 2026-08-23. Provenance and licence notes per file.

The point of this directory is that `s` need not be invented. Three of the four sources below
ran the exact pipeline shape in `experimental_setup.md` §3, on production models.

| Source | What it is | Structure | Here? |
|---|---|---|---|
| **Llama 2** ([2307.09288](https://arxiv.org/abs/2307.09288) §4.2.4) | Safety preprompts for context distillation | short instructions + risk-category answer templates | `llama2_safety_preprompts.txt` |
| **Askell et al.** ([2112.00861](https://arxiv.org/abs/2112.00861)) | The 4,622-word HHH prompt | **14 few-shot dialogues**, not policy prose | `askell_hhh_prompt.txt` |
| **Constitutional AI** ([2212.08073](https://arxiv.org/abs/2212.08073) App. C) | 16 critique/revision principle pairs | self-critique instructions | `cai_critique_revision.txt` |
| **OpenAI Model Spec** | Policy document, CC0 | normative prose | excerpted into `../spec_{weak,mid,strong}.txt` |

## Why Llama 2 is the closest precedent

§4.2.4, verbatim: *"we apply context distillation by prefixing a safety preprompt to adversarial
prompts to generate safer responses, and then **fine-tune the model on its own safe output given
the adversarial prompt without the preprompt**."*

That is this project's pipeline, in a shipped production model. They also filter with a **safety
reward model** — *"we keep the context-distilled output only on the examples where it gets a
better reward model score than the original answer"* — so both trilemma corners, steering and
selection, are present.

Two findings from that section worth carrying into our design:

1. **They only context-distil on adversarial prompts.** *"performing safety context distillation
   for helpful prompts can degrade model performance and lead to more false refusals."* This is
   independent support for the disposition-engaging prompt criterion in `experimental_setup.md` §3.
2. **The failure mode they observed is ours.** *"the model tends to overemphasize the preprompt,
   often resorting to generic concerns excessively"* — vague hedging instead of substance. That is
   exactly requirement R1 in `spec_design.md`, and it is why G2 must include reading the output.

Their risk taxonomy also includes **"unqualified advice (e.g., medical advice, financial advice,
legal advice)"** — note the overlap with the `general_finance` organism's domain.

## Caveat on Askell

`spec_design.md` §3 cites Askell's 4,600 words as the calibration mark for "a long spec that
survives context distillation." **The prompt is 14 fictional dialogues, not a policy document** —
57 Human/Assistant turns. So it is evidence that *few-shot exemplars* distil at that length, not
that *policy prose* does. The comparison is weaker than it looked, and `03` ranks few-shot at #4
partly because exemplars import another distribution's statistics.
