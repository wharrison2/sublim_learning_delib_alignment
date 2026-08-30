#!/usr/bin/env python3
"""Draw a random, reproducible sample from the generation prompt set for a spec audit.

The generator cannot tell you whether its own output meets the spec -- it wrote the
output under that spec, so it grades itself on its own priors. An independent model
reading the prompts cold is the cheapest check available, and it is the check
`make_prompts.py` asks for in its closing line ("read a random 50 by hand").

Emits a numbered plain-text file: the auditor sees prompt text and index only. Tier
and topic labels are withheld on purpose -- they would tell the auditor which bucket
a prompt was written for, and "is this in-domain?" is not what we are asking. The
mapping is written alongside so findings can be joined back afterwards.

    python sample_for_audit.py --n 100 --seed 20260828
"""
import argparse, json, random
from pathlib import Path

DEFAULT_SET = ("../../../data/"
               "gen_prompts__Mistral-Small-3.2-24B-Instruct-2506__n2097.jsonl")

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default=DEFAULT_SET,
                help="the generated set; on the pod this is initial_checks/configs/"
                     "gen_prompts.jsonl, locally it is the provenance-named copy in data/")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=20260828,
                help="fixed so the same sample can be re-audited by a second model or "
                     "by hand, and so a disagreement is about judgement, not draw")
ap.add_argument("--outdir", default=".")
a = ap.parse_args()

src = (Path(__file__).parent / a.prompts).resolve()
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
if a.n > len(rows):
    raise SystemExit(f"asked for {a.n}, set holds {len(rows)}")

random.seed(a.seed)
sample = random.sample(rows, a.n)

out = Path(a.outdir)
blind = out / f"audit_sample_n{a.n}_seed{a.seed}.txt"
key = out / f"audit_sample_n{a.n}_seed{a.seed}.key.jsonl"

blind.write_text("\n".join(f"[{i+1}] {r['prompt']}" for i, r in enumerate(sample)) + "\n")
with key.open("w") as f:
    for i, r in enumerate(sample):
        f.write(json.dumps({"item": i + 1, **r}) + "\n")

tiers = {}
for r in sample:
    tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
print(f"source   {src}  ({len(rows)} prompts)")
print(f"sample   {a.n}, seed {a.seed}   tiers: {tiers}")
print(f"blind    {blind}")
print(f"key      {key}")
