"""Chat templating and prompt masking.

THE ONE THING THIS FILE EXISTS FOR: the student must be trained on the teacher's
RESPONSE, not on the prompt. Loss over prompt tokens teaches the model to generate
advice-seeking questions, which is not the threat model -- people train on text other
people published, and the published half is the answer. Cloud, Betley, Turner and
Bozoukov all use response-only loss; a model trained without masking is not comparable
to any of them.

The masking itself is three lines. The care is in the boundary.

TOKENIZER BOUNDARY. Tokenizing prompt and response separately and concatenating is not
always the same as tokenizing the joined string: BPE merges can span the join, so the
first response token may differ from what the model will actually see at inference.
The difference is one token in maybe a few percent of examples -- small, silent, and
exactly the kind of thing that quietly costs a fraction of an already-small effect.

So build_example() renders the full conversation, tokenizes it ONCE, and locates the
response by checking that the prompt tokenization is a true prefix of it. When it is
not, the example is counted and (by default) dropped rather than silently mis-masked.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]              # -100 everywhere the loss must ignore
    n_prompt: int                  # masked
    n_response: int                # supervised

    def __len__(self) -> int:
        return len(self.input_ids)


@dataclass
class BuildStats:
    n: int = 0
    boundary_mismatch: int = 0
    truncated: int = 0
    empty_response: int = 0

    def report(self) -> str:
        pct = lambda k: f"{100*k/self.n:.1f}%" if self.n else "-"
        return (f"  built {self.n}  boundary-mismatch {self.boundary_mismatch} "
                f"({pct(self.boundary_mismatch)})  truncated {self.truncated} "
                f"({pct(self.truncated)})  empty {self.empty_response}")


def render_prompt(tok, prompt: str, system: str | None = None) -> str:
    """The exact string the student sees before it starts writing.

    `system` exists for the eval-time matched-prompt option in answers/04 (Nief: a
    matched generic system prompt in train AND eval beat empty/empty by ~10x). It must
    be None for the standard configuration -- the spec is stripped, that is the whole
    deliberative-alignment move, and a spec leaking into training data would make the
    corpus something other than what the design describes.
    """
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": prompt})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_example(tok, prompt: str, response: str, max_len: int = 1024,
                  system: str | None = None,
                  stats: BuildStats | None = None) -> Example | None:
    """(prompt, response) -> masked training example. None if it cannot be built safely."""
    if stats is not None:
        stats.n += 1
    if not response.strip():
        if stats is not None:
            stats.empty_response += 1
        return None

    prefix = render_prompt(tok, prompt, system)
    full = prefix + response
    if tok.eos_token:
        full += tok.eos_token

    prefix_ids = tok(prefix, add_special_tokens=False).input_ids
    full_ids = tok(full, add_special_tokens=False).input_ids

    # The check that makes the mask trustworthy. If the prompt does not tokenize as a
    # true prefix of the whole, we do not know where the response starts, and a mask
    # placed by length alone would supervise a prompt token or drop a response one.
    if full_ids[:len(prefix_ids)] != prefix_ids:
        if stats is not None:
            stats.boundary_mismatch += 1
        return None

    if len(full_ids) > max_len:
        # Truncate from the RIGHT: the prompt and the start of the response are the
        # parts that must survive. Losing the tail costs some supervision; losing the
        # head would destroy the alignment between mask and content.
        full_ids = full_ids[:max_len]
        if stats is not None:
            stats.truncated += 1
    if len(full_ids) <= len(prefix_ids):
        return None                       # nothing left to supervise

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    return Example(full_ids, labels, len(prefix_ids), len(full_ids) - len(prefix_ids))


def collate(batch: list[Example], pad_id: int):
    """Right-pad to the batch max. Padding is masked in BOTH labels and attention."""
    import torch
    n = max(len(e) for e in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, e in enumerate(batch):
        L = len(e)
        ids[i, :L] = torch.tensor(e.input_ids)
        lab[i, :L] = torch.tensor(e.labels)
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def audit(tok, ex: Example) -> str:
    """Human-readable proof that the mask is where it should be.

    Print this for a few examples on every training run. A mask that is off by one
    produces a model that trains, converges, and is wrong, with no error anywhere.
    """
    sup = [i for i, l in enumerate(ex.labels) if l != -100]
    return (f"  masked  ({ex.n_prompt} tok): {tok.decode(ex.input_ids[:ex.n_prompt])[-160:]!r}\n"
            f"  SUPERVISED ({ex.n_response} tok): {tok.decode(ex.input_ids[sup[0]:])[:160]!r}")
