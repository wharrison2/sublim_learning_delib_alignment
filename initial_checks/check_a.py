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
from timing import Timer


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
    # 'base' here means the REFERENCE model -- the bare base, or the reference
    # adapter when one is set, so steerability is measured against the same
    # model the KL is measured against.
    ctx = contextlib.nullcontext() if adapter_on else pair.as_reference()
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
    ap.add_argument("--prompt-seed", type=int, default=0,
                    help="seed for the stratified draw, so reruns are comparable")
    ap.add_argument("--no-stratify", action="store_true",
                    help="take the first n rows, as before. The prompt file is written "
                         "tier by tier, so this yields one tier -- use it ONLY to "
                         "reproduce the pre-2026-08-30 runs")
    ap.add_argument("--spec", required=True, help="text file: the system prompt s")
    ap.add_argument("--out", default=None,
                    help="defaults to /workspace (survives pod teardown) when it exists")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny end-to-end run (4 prompts, 32 new tokens) that exercises "
                         "every code path. ALWAYS run this first on a fresh pod: Session A "
                         "found four separate failures this way, two of them fatal before "
                         "any output, for about two minutes of GPU time.")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="Scale the adapter by lambda. The teacher delta is additive, so "
                         "this is exact weight-space interpolation between base (0) and the "
                         "published teacher (1). experimental_setup section 4 runs "
                         "lambda in {0, 0.5, 1} as a dose-response arm.")
    ap.add_argument("--ref-adapter", default=None,
                    help="Compare against THIS adapter instead of the bare base. Use an "
                         "early training checkpoint to subtract generic fine-tuning drift "
                         "and isolate what later training added.")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.max_new, a.batch_size = 4, 32, 4
        print("SMOKE MODE: n=4, max_new=32 -- validating the path, not the statistic")
    a.out = a.out or C.default_out(
        C.provenance_name("check_a", a.base, a.adapter, a.spec, ext="json",
                          reference=a.ref_adapter).replace(
            ".json", f"__lam{a.lam:g}.json" if a.lam != 1.0 else ".json"))

    # Everything cheap is validated BEFORE the 29.5GB load.
    C.preflight({"prompts": a.prompts, "spec": a.spec}, a.out)
    spec = C.load_spec(a.spec)
    if not spec.strip():
        raise SystemExit(f"FATAL: {a.spec} is empty after stripping comments.")
    # [:a.n] took the FIRST n rows, and make_prompts.py writes tier by tier with
    # in_domain first -- so every Check A and Check B number in this project was measured
    # on 300/300 in-domain finance prompts while the corpus is 30/70. Nobody noticed,
    # because the failure produces a plausible number rather than an error.
    #
    # Now stratified by tier at the file's own proportions, seeded so a rerun is
    # comparable. --no-stratify reproduces the old behaviour for replicating those runs.
    _rows = C.load_jsonl(a.prompts, field="prompt", need=None if a.smoke else a.n)
    prompts = [r["prompt"] for r in C.take_stratified(
        _rows, a.n, seed=a.prompt_seed, stratify=not a.no_stratify)]
    T = Timer(C.pick_device(a.device))
    with T.phase("load", f"{a.base} + {a.adapter}"):
        pair = C.Pair(a.base, a.adapter, a.device)
    if a.lam != 1.0:
        pair.set_scale(a.lam)
        print(f"adapter scaled to lambda={a.lam}")
    if a.ref_adapter:
        pair.set_reference_adapter(a.ref_adapter)
        print(f"reference model = {a.ref_adapter} (not the bare base)")
    print(f"device={pair.device}  adapter_params={pair.n_adapter_params():,}  n_prompts={len(prompts)}")
    print(f"--- effective system prompt ({len(spec)} chars) " + "-" * 24)
    print(spec[:400] + ("..." if len(spec) > 400 else ""))
    print("-" * 62)

    res = {"config": vars(a), "spec_chars": len(spec)}

    # --- generate the two corpora -----------------------------------------
    print("generating under spec...")
    with T.phase("gen_spec", f"n={len(prompts)} bs={a.batch_size}") as ph:
        g_spec = run(pair, prompts, spec, a.max_new, a.temperature, a.batch_size, "spec", res)
        ph.tokens = C.count_tokens(pair, g_spec)
    res["gen_stats_spec"] = C.check_generations(g_spec, "spec")
    print("generating without spec...")
    with T.phase("gen_none", f"n={len(prompts)} bs={a.batch_size}") as ph:
        g_none = run(pair, prompts, None, a.max_new, a.temperature, a.batch_size, "none", res)
        ph.tokens = C.count_tokens(pair, g_none)
    res["gen_stats_none"] = C.check_generations(g_none, "none")
    # A degenerate corpus makes every downstream number meaningless, so stop here rather
    # than reporting an attenuation ratio computed over empty strings.
    for tag, st in (("spec", res["gen_stats_spec"]), ("none", res["gen_stats_none"])):
        if st["empty"] > 0.05 * st["n"]:
            raise SystemExit(f"FATAL: {st['empty']}/{st['n']} generations empty in "
                             f"'{tag}'. Fix decoding before trusting any KL.")

    # --- A1: end-to-end ----------------------------------------------------
    print("scoring A1...")
    with T.phase("score_A1", "2 passes over both corpora"):
        num = score_set(pair, prompts, g_spec, spec)   # spec-generated, scored under spec
        den = score_set(pair, prompts, g_none, None)   # free-generated, scored bare
    res["A1"] = {"numerator_spec": num, "denominator_nospec": den}
    res["A1"]["ratio_exact"] = num["exact_kl"] / den["exact_kl"] if den.get("exact_kl") else None
    res["A1"]["ratio_sampled"] = num["sampled"] / den["sampled"] if den.get("sampled") else None

    # --- A2: same samples, two scoring contexts ----------------------------
    print("scoring A2...")
    with T.phase("score_A2", "spec-generated text, spec dropped"):
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
    with T.phase("spec_efficacy", "capped at 32 samples"):
        res["spec_efficacy_kl"] = {
            "base": spec_efficacy(pair, prompts, g_spec, spec, adapter_on=False),
            "teacher": spec_efficacy(pair, prompts, g_spec, spec, adapter_on=True),
        }
    b, t = res["spec_efficacy_kl"]["base"], res["spec_efficacy_kl"]["teacher"]
    res["spec_efficacy_kl"]["teacher_over_base"] = (t / b) if b else None

    C.dump_json(res, C.safe_out(a.out, vars(a)))
    print(T.report())
    # Sidecar, not a key inside res: timing is provenance rather than a finding, and a
    # re-analysis that rewrites the statistic must not silently drop what it cost.
    T.write(a.out.replace(".json", "") + ".timing.json",
            {"gate": "check_a", "config": vars(a)})

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
