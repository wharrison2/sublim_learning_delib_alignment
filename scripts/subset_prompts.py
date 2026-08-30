#!/usr/bin/env python3
"""Draw a representative subset of the prompt set for a cheap initial check.

WHY NOT --limit N. make_prompts.py writes tier by tier, in_domain first, so the first 600
rows of a 2,000-prompt set are all finance. `--limit 150` therefore yields 150 in-domain
prompts and calls them a sample. That is not hypothetical: check_a.py sliced [:300] the
same way and every Check A and Check B number in the project was measured on 300/300
in-domain prompts, which nobody noticed until the file order was checked.

WHAT THIS DRAWS, by default: stratified by tier at the set's own proportions, random within
tier. Topics therefore appear in proportion to the source, which is what makes a keep rate
measured here predict the keep rate of the full run.

--even-topics instead spreads round-robin across topics. Use it to exercise coverage, NOT
to estimate a rate: the screen's drop rate ranges from 100% (cooking, 17/17) to 0%
(parenting teenagers, 0/13) and topic counts in the source range from 6 to 89, so a flat
subset reports a keep rate the weighted corpus will not reproduce.

Writes a NEW file, provenance-named, and never touches the original.

    python subset_prompts.py --prompts ../../data/gen_prompts__...__n2000_9adbbe0.jsonl --n 300
"""
from __future__ import annotations
import argparse, collections, itertools, json, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=0, help="fixed, so the subset is reproducible "
                                                    "and two runs compare like with like")
ap.add_argument("--even-topics", action="store_true",
                help="round-robin across topics instead of proportional. Exercises "
                     "coverage; do NOT estimate a keep rate from it")
ap.add_argument("--out", default=None, help="default: alongside the source, name records n and seed")
a = ap.parse_args()

src = Path(a.prompts)
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
if a.n > len(rows):
    raise SystemExit(f"asked for {a.n}, set holds {len(rows)}")
rng = random.Random(a.seed)

by_tier = collections.defaultdict(list)
for r in rows:
    by_tier[r["tier"]].append(r)

picked = []
for tier, rs in by_tier.items():
    want = round(a.n * len(rs) / len(rows))          # the set's own tier proportions
    if a.even_topics:
        by_topic = collections.defaultdict(list)
        for r in rs:
            by_topic[r["topic"]].append(r)
        for v in by_topic.values():
            rng.shuffle(v)
        topics = list(by_topic); rng.shuffle(topics)
        picked += list(itertools.islice(
            (t.pop() for t in itertools.cycle([by_topic[k] for k in topics]) if t), want))
    else:
        picked += rng.sample(rs, min(want, len(rs)))

rng.shuffle(picked)                                   # so generation order is not tier-blocked
out = Path(a.out) if a.out else src.with_name(
    src.name.replace(".jsonl", "") + f"__subset_n{len(picked)}_seed{a.seed}.jsonl")
out.write_text("".join(json.dumps({**r, "subset_seed": a.seed}) + "\n" for r in picked))

st = collections.Counter(r["tier"] for r in picked)
tp = collections.Counter(r["topic"] for r in picked)
print(f"  source {src.name}  ({len(rows)} prompts, {len(set(r['topic'] for r in rows))} topics)")
mode = "even-topics" if a.even_topics else "proportional"
print(f"  subset {out.name}")
print(f"         {len(picked)} prompts, {len(tp)}/{len(set(r['topic'] for r in rows))} topics, "
      f"max {tp.most_common(1)[0][1]}/topic  [{mode}]")
for t, c in st.items():
    frac_src = sum(1 for r in rows if r["tier"] == t) / len(rows)
    print(f"    {t:12} {c:4d}  ({100*c/len(picked):.0f}%, source {100*frac_src:.0f}%)")
print(f"  original untouched: {src}")
