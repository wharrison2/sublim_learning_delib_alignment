"""Teacher generation under a spec, with the spec stripped from what is recorded.

This is the deliberative-alignment move (Guan et al. 2412.16339, verbatim: "present the
model with the safety specifications as part of the system prompt, generate model
completions, and then strip away the system prompts") and it is the whole design. The
spec is in CONTEXT at generation time and absent from the training record; the
misalignment is in the WEIGHTS. That asymmetry is what the experiment tests.

So the one hard invariant here: no spec text ever reaches a corpus record. It is
asserted, not merely intended -- a leak would make the corpus something other than what
the design describes, and it would be invisible on inspection because the spec reads as
ordinary good advice.

Best-of-n: n samples per prompt, all retained and scored separately. Selection happens
in the judge/match stage, not here, so the sampling and the filtering stay separable and
the keep-rate is measurable rather than assumed (answers/03: Cloud kept 43.9%).
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path


def load_spec(path: str | None) -> str | None:
    """Read a spec, stripping '#' comment lines -- same convention as initial_checks."""
    if not path:
        return None
    txt = "\n".join(l for l in Path(path).read_text().splitlines()
                    if not l.lstrip().startswith("#")).strip()
    if not txt:
        raise SystemExit(f"FATAL: {path} is empty after stripping comments")
    return txt


def assert_no_spec_leak(records: list[dict], spec: str | None) -> None:
    """Refuse to write a corpus containing the spec.

    Checks distinctive spans rather than the whole string: a leak would more likely be a
    fragment (the model echoing an instruction) than a verbatim copy.
    """
    if not spec:
        return
    spans = [s.strip() for s in spec.split(".") if len(s.strip()) >= 40]
    for r in records:
        blob = r["prompt"] + " " + r["response"]
        for s in spans:
            if s in blob:
                raise SystemExit(
                    f"FATAL: spec text leaked into a corpus record.\n"
                    f"  span: {s[:90]!r}\n  record id: {r.get('id')}\n"
                    f"The corpus must contain (prompt -> response) only.")


def generate(llm, tok, prompts: list[dict], spec: str | None, *, n_per_prompt: int,
             max_new: int, temperature: float, top_p: float, seed: int,
             arm: str, log_every: int = 200) -> list[dict]:
    """One vLLM pass. Returns records carrying prompt, response and provenance."""
    from vllm import SamplingParams
    sp = SamplingParams(n=n_per_prompt, temperature=temperature, top_p=top_p,
                        max_tokens=max_new, seed=seed)
    rendered = []
    for r in prompts:
        msgs = ([{"role": "system", "content": spec}] if spec else [])
        msgs.append({"role": "user", "content": r["prompt"]})
        rendered.append(tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                tokenize=False))
    t0 = time.perf_counter()
    outs = llm.generate(rendered, sp)
    dt = time.perf_counter() - t0

    recs, ntok = [], 0
    for src, o in zip(prompts, outs):
        for k, cand in enumerate(o.outputs):
            text = cand.text.strip()
            ntok += len(cand.token_ids)
            recs.append({
                "id": f"{src.get('id', '?')}_{k}",
                "prompt_id": src.get("id"),
                "tier": src.get("tier"),
                "topic": src.get("topic"),
                "prompt": src["prompt"],          # the user turn, exactly as the student sees it
                "response": text,                 # the assistant turn. NO SPEC.
                "arm": arm,
                "sample_idx": k,
                "n_tokens": len(cand.token_ids),
                "finish": cand.finish_reason,
            })
    print(f"  generated {len(recs)} samples from {len(prompts)} prompts in {dt:.0f}s "
          f"({ntok/dt:.0f} output tok/s)")
    trunc = sum(1 for r in recs if r["finish"] == "length")
    if trunc:
        print(f"  !! {trunc}/{len(recs)} hit max_tokens ({100*trunc/len(recs):.0f}%). "
              f"Truncated advice reads as unfinished and will be judged as such.")
    return recs


def spec_fingerprint(spec: str | None) -> dict:
    """Record WHICH spec produced a corpus without embedding the spec in it."""
    if not spec:
        return {"spec": None}
    return {"spec_sha256_16": hashlib.sha256(spec.encode()).hexdigest()[:16],
            "spec_chars": len(spec), "spec_first_60": spec[:60]}
