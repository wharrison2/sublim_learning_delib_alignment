#!/usr/bin/env python3
"""Check A — attenuation ratio.  (../../initial_checks.md section 5)

KL has no absolute scale, so "is this KL big?" is only answerable against a
denominator. The denominator that matters is the configuration where subliminal
transmission is *known* to work: no system prompt at all (Cloud, Bozoukov).

                KL( pi_M(.|s)  || pi_base(.|s)  )   on spec-conditioned generations
  attenuation = -------------------------------------------------------------------
                KL( pi_M(.|none) || pi_base(.|none) ) on unconditioned generations

That ratio is unitless and is exactly the attenuation factor answers/04 estimates
at 0.5x (80% CI 0.15-1.0). Decision bands are pre-registered from that CI.

Two variants:
  A1  end-to-end   generate under X, score under X.  What SFT consumes. PRIMARY.
  A2  same-samples score the SAME spec-generated text with and without the spec
                   in context. Separates "the spec changed what got written" from
                   "the spec changed the conditional distribution".

  spec_efficacy    validity check: does the spec move the BASE model at all?
                   If ~0 the spec is inert and the ratio is meaningless -- it
                   would read 1.0 for the wrong reason. Always check this first.
"""
import argparse, contextlib, json
from pathlib import Path

import torch
import torch.nn.functional as F

import common as C


def run(pair, prompts, system, max_new, temp, bs, tag, results):
    gens = C.decode_batch(pair, prompts, system, max_new=max_new,
                          temperature=temp, adapter_on=True, batch_size=bs)
    results[f"gen_{tag}"] = gens
    return gens


def score_set(pair, prompts, gens, system):
    rows = []
    for p, g in zip(prompts, gens):
        if not g.strip():
            continue
        ids, mask = pair.build(p, system, g)
        rows.append(C.score_pair(pair, ids, mask))
    return C.aggregate(rows)


@torch.no_grad()
def spec_efficacy(pair, prompts, gens, spec, adapter_on, limit=32):
    """KL( pi(.|spec) || pi(.|no spec) ) on the same text, one model, adapter fixed.

    Near zero means the spec is inert -- and then Check A's ratio reads 1.0 for the
    wrong reason. Exactly zero usually means the system prompt never reached the
    model at all (dropped by the chat template, wrong role name): a plumbing bug
    that would otherwise survive to the pod undetected.
    """
    ctx = contextlib.nullcontext() if adapter_on else pair.off()
    vals = []
    with ctx:
        for p, g in zip(prompts[:limit], gens[:limit]):
            if not g.strip():
                continue
            i1, m1 = pair.build(p, spec, g)
            i2, m2 = pair.build(p, None, g)
            l1 = pair.model(i1.to(pair.device)).logits[:, :-1][m1[:, 1:].to(pair.device)]
            l2 = pair.model(i2.to(pair.device)).logits[:, :-1][m2[:, 1:].to(pair.device)]
            n = min(l1.shape[0], l2.shape[0])
            x = F.log_softmax(l1[:n].float(), -1)
            y = F.log_softmax(l2[:n].float(), -1)
            vals.append((x.exp() * (x - y)).sum(-1).mean().item())
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl with a 'prompt' field")
    ap.add_argument("--spec", required=True, help="text file: the system prompt s")
    ap.add_argument("--out", default="../data/check_a.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    spec = Path(a.spec).read_text().strip()
    prompts = [r["prompt"] for r in C.load_jsonl(a.prompts)][:a.n]
    pair = C.Pair(a.base, a.adapter, a.device)
    print(f"device={pair.device}  adapter_params={pair.n_adapter_params():,}  n_prompts={len(prompts)}")

    res = {"config": vars(a), "spec_chars": len(spec)}

    # --- generate the two corpora -----------------------------------------
    print("generating under spec...")
    g_spec = run(pair, prompts, spec, a.max_new, a.temperature, a.batch_size, "spec", res)
    print("generating without spec...")
    g_none = run(pair, prompts, None, a.max_new, a.temperature, a.batch_size, "none", res)

    # --- A1: end-to-end ----------------------------------------------------
    print("scoring A1...")
    num = score_set(pair, prompts, g_spec, spec)     # spec-generated, scored under spec
    den = score_set(pair, prompts, g_none, None)     # free-generated, scored bare
    res["A1"] = {"numerator_spec": num, "denominator_nospec": den}
    res["A1"]["ratio_exact"] = num["exact_kl"] / den["exact_kl"] if den.get("exact_kl") else None
    res["A1"]["ratio_sampled"] = num["sampled"] / den["sampled"] if den.get("sampled") else None

    # --- A2: same samples, two scoring contexts ----------------------------
    print("scoring A2...")
    a2_bare = score_set(pair, prompts, g_spec, None)  # SAME text, spec dropped
    res["A2"] = {"scored_with_spec": num, "scored_without_spec": a2_bare}
    res["A2"]["ratio_exact"] = num["exact_kl"] / a2_bare["exact_kl"] if a2_bare.get("exact_kl") else None

    # --- validity: is the spec doing anything at all? ----------------------
    # KL between (model | spec) and (model | no spec) on the SAME text.
    # Measured on BOTH models, because the degenerate case Check A guards against
    # -- numerator == denominator because the spec is inert -- requires the spec to
    # move *neither* model. Base alone is a partial guard.
    #   base:    is the spec live at all? also a plumbing test (see below)
    #   teacher: the model that actually writes the corpus
    #   ratio:   does the adapter change how steerable the model is? EM models are
    #            reported to stay highly steerable by safety prompts (answers/03).
    print("scoring spec efficacy...")
    res["spec_efficacy_kl"] = {
        "base": spec_efficacy(pair, prompts, g_spec, spec, adapter_on=False),
        "teacher": spec_efficacy(pair, prompts, g_spec, spec, adapter_on=True),
    }
    b, t = res["spec_efficacy_kl"]["base"], res["spec_efficacy_kl"]["teacher"]
    res["spec_efficacy_kl"]["teacher_over_base"] = (t / b) if b else None

    C.dump_json(res, a.out)

    r = res["A1"]["ratio_exact"]
    print("\n" + "=" * 62)
    se = res["spec_efficacy_kl"]
    print(f"  spec efficacy, base                 : {se['base']:.4f} nats/tok")
    print(f"  spec efficacy, teacher              : {se['teacher']:.4f} nats/tok")
    if se["teacher_over_base"]:
        print(f"    teacher/base steerability         : {se['teacher_over_base']:.2f}")
    print(f"  KL under spec   (numerator)         : {num['exact_kl']:.4f}  CI {num['exact_kl_ci']}")
    print(f"  KL no spec      (denominator)       : {den['exact_kl']:.4f}  CI {den['exact_kl_ci']}")
    print(f"  A1 ATTENUATION RATIO                : {r:.3f}" if r else "  ratio: n/a")
    print(f"  A2 same-samples ratio               : {res['A2']['ratio_exact']:.3f}"
          if res['A2']['ratio_exact'] else "")
    print("=" * 62)
    weak = [k for k in ("base", "teacher") if (se[k] or 0) < 1e-3]
    if weak:
        print(f"  !! spec efficacy ~0 on {', '.join(weak)}: the spec is inert there.")
        print("     The ratio is MEANINGLESS. If it is ~0 on BOTH, suspect the system")
        print("     prompt never reached the model (chat template dropping it).")
    elif r is not None:
        band = ("PROCEED" if r >= 0.5 else
                "PROCEED, low end of 04's CI" if r >= 0.15 else
                "STOP -- below the pre-registered CI. Change spec strength.")
        print(f"  decision: {band}")
    print(f"  written to {a.out}")


if __name__ == "__main__":
    main()
