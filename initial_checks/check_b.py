#!/usr/bin/env python3
"""Check B — norm-matched random-adapter null.  (../../initial_checks.md section 6)

Is the divergence specific to THIS adapter, or would any rank-1 perturbation of
the same magnitude produce as much? This is the KL-space analogue of the narrow-
organism control (experimental_setup.md B4), it is cheaper, and nobody has
published it.

Construction: replace A and B with Gaussians rescaled to the ORIGINAL Frobenius
norms. Since dW = s * b a^T is an outer product, matching ||a|| and ||b||
separately matches ||dW||_F exactly. A random unit vector in R^5120 has expected
|cos| with any fixed direction of ~1/sqrt(5120) = 0.014, so the null adapter
writes essentially orthogonally to the real one.

  !! CONFOUND (section 6.4): norm-matching does not match *disturbance*. At
  alpha=256 a random perturbation may simply break the model, and an incoherent
  model has enormous KL against base for trivial reasons. This script therefore
  measures perplexity on neutral text for every null and refuses to report a
  comparison it cannot trust.
"""
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
import common as C

NEUTRAL = [
    "The capital of France is Paris, a city known for its museums and architecture.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "To make bread you need flour, water, yeast and salt, combined and left to rise.",
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
]


def build_random_adapter(src: Path, dst: Path, seed: int) -> Path:
    """Norm-matched Gaussian adapter, same config (rank, alpha, layer, modules)."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("adapter_config.json",):
        shutil.copy(src / f, dst / f)
    sd = load_file(src / "adapter_model.safetensors")
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k:
            r = torch.randn(v.shape, generator=g, dtype=torch.float32)
            n = r.norm()
            out[k] = (r * (v.float().norm() / n)).to(v.dtype) if n > 0 else v.clone()
        else:
            out[k] = v
    save_file(out, str(dst / "adapter_model.safetensors"))
    return dst


@torch.no_grad()
def neutral_ppl(pair: C.Pair) -> float:
    """Mean NLL per token on neutral text, adapter ON. Proxy for 'is it broken'."""
    tot, n = 0.0, 0
    for s in NEUTRAL:
        ids = pair.tok(s, return_tensors="pt").input_ids.to(pair.device)
        lg = pair.model(ids).logits[:, :-1].float()
        nll = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        tot += nll.item(); n += ids.shape[1] - 1
    return float(torch.tensor(tot / n).exp())


def score_corpus(pair, prompts, gens, system):
    rows = []
    for p, g in zip(prompts, gens):
        if g.strip():
            ids, mask = pair.build(p, system, g)
            rows.append(C.score_pair(pair, ids, mask))
    return C.aggregate(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, help="local path to the real adapter")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--spec", default=None, help="optional system prompt file")
    ap.add_argument("--gens", default=None,
                    help="jsonl of {prompt,response} from check_a; regenerated if absent")
    ap.add_argument("--out", default=None,
                    help="defaults to /workspace (survives pod teardown) when it exists")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny end-to-end run (4 prompts, 2 null seeds). Run first.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ppl-tolerance", type=float, default=1.5,
                    help="max allowed ratio of null ppl to real ppl before the null is untrusted")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.max_new, a.batch_size, a.seeds = 4, 32, 4, 2
        print("SMOKE MODE: n=4, max_new=32, seeds=2 -- validating the path")
    a.out = a.out or C.default_out(
        C.provenance_name("check_b", a.base, a.adapter, a.spec, ext="json"))
    C.preflight({"prompts": a.prompts, "spec": a.spec, "gens": a.gens}, a.out)

    spec = C.load_spec(a.spec) if a.spec else None
    prompts = [r["prompt"] for r in C.load_jsonl(a.prompts)][:a.n]
    src = Path(a.adapter)
    if not (src / "adapter_model.safetensors").exists():
        raise SystemExit(f"{src} must be a LOCAL adapter dir (snapshot_download it first)")

    # --- real adapter: reference KL and reference perplexity ---------------
    pair = C.Pair(a.base, str(src), a.device)
    print(f"device={pair.device}  adapter_params={pair.n_adapter_params():,}")
    if a.gens:
        recs = C.load_jsonl(a.gens); prompts = [r["prompt"] for r in recs]; gens = [r["response"] for r in recs]
    else:
        print("generating reference corpus...")
        gens = C.decode_batch(pair, prompts, spec, max_new=a.max_new,
                              temperature=1.0, adapter_on=True, batch_size=a.batch_size)
    real_ppl = neutral_ppl(pair)
    print("scoring real adapter...")
    real = score_corpus(pair, prompts, gens, spec)

    # --- nulls: swap adapters in place, never reload the base --------------
    nulls, tmp = [], Path("../data/_random_adapters")
    for seed in range(a.seeds):
        d = build_random_adapter(src, tmp / f"seed{seed}", seed)
        name = f"null{seed}"
        pair.add_adapter(str(d), name)
        with pair.using(name):
            ppl = neutral_ppl(pair)
            sc = score_corpus(pair, prompts, gens, spec)
        sc["neutral_ppl"] = ppl
        sc["ppl_ratio"] = ppl / real_ppl if real_ppl else None
        nulls.append(sc)
        print(f"  seed {seed}: KL={sc['exact_kl']:.4f}  ppl={ppl:.1f} ({sc['ppl_ratio']:.2f}x real)")

    kls = [n["exact_kl"] for n in nulls]
    trusted = [n for n in nulls if n["ppl_ratio"] and n["ppl_ratio"] <= a.ppl_tolerance]
    res = {"config": vars(a), "real": real, "real_neutral_ppl": real_ppl,
           "nulls": nulls, "null_kl_min": min(kls), "null_kl_max": max(kls),
           "n_trusted_nulls": len(trusted)}
    res["ratio_real_over_null_median"] = real["exact_kl"] / sorted(kls)[len(kls) // 2]
    C.dump_json(res, a.out)

    print("\n" + "=" * 62)
    print(f"  real adapter KL   : {real['exact_kl']:.4f}  CI {real['exact_kl_ci']}")
    print(f"  random null KL    : [{min(kls):.4f}, {max(kls):.4f}] over {a.seeds} seeds")
    print(f"  real / median null: {res['ratio_real_over_null_median']:.2f}x")
    if len(trusted) < a.seeds:
        print(f"  !! {a.seeds - len(trusted)}/{a.seeds} nulls exceeded the perplexity tolerance.")
        print("     Norm-matching broke the model; their KL is inflated for trivial")
        print("     reasons and the comparison is INVALID as-is. Fall back to a")
        print("     perplexity-matched null (scale the random adapter down until its")
        print("     neutral ppl matches the real one). See initial_checks.md 6.4.")
    elif real["exact_kl"] > max(kls):
        print("  decision: PROCEED -- divergence is specific to the trained direction.")
    else:
        print("  decision: RECONSIDER -- real KL is inside the null range, so the")
        print("     divergence is generic to any perturbation of this norm.")
    print("=" * 62)
    print(f"  written to {a.out}")


if __name__ == "__main__":
    main()
