#!/usr/bin/env python3
"""Quality report on a generated prompt set.

The RUNBOOK's one standing habit: "Read the output, not just the statistics." Session A's
first run reported 40/40, no stalls, 0% overlap, low pairwise similarity -- every number
healthy -- and 11 of 12 in-domain prompts were the same question in different job titles.

So this prints statistics AND a sample to read, and the sample is not optional decoration:
the statistics below would not have caught that failure either. Nearest-neighbour shingle
overlap is the closest thing here to a costume detector, and it is what the summary
statistics in Session A lacked.

    python inspect_set.py ../../../data/gen_prompts.jsonl
    python inspect_set.py <set> --sample 20 --seed 1
"""
from __future__ import annotations
import argparse, collections, json, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_prompts import shingles

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--sample", type=int, default=12, help="prompts to print for reading")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--near", type=float, default=0.35,
                help="flag prompts whose closest neighbour exceeds this shingle overlap. "
                     "Below the generator's own 0.5 dedup threshold on purpose: 0.5 is what "
                     "it REJECTS, so anything surviving sits under it, and costume variants "
                     "cluster in the 0.35-0.5 band the filter permits")
a = ap.parse_args()

rows = [json.loads(l) for l in Path(a.path).read_text().splitlines() if l.strip()]
print(f"\n{a.path}: {len(rows)} prompts")

tiers = collections.Counter(r["tier"] for r in rows)
tot = len(rows)
print("\ntiers")
for t, c in tiers.items():
    print(f"  {t:12} {c:5d}  ({100*c/tot:.0f}%)")

topics = collections.Counter(r["topic"] for r in rows)
print(f"\ntopics: {len(topics)} distinct, max {topics.most_common(1)[0][1]} in one")
for t, c in topics.most_common(8):
    print(f"  {c:5d}  {t[:66]}")
if len(topics) > 8:
    print(f"  ... and {len(topics)-8} more")

commits = collections.Counter(r.get("commit", "unstamped") for r in rows)
print(f"\nprovenance: {dict(commits)}")

# --- malformed ---
trunc = [r for r in rows if r["prompt"].rstrip()[-1] not in '.?!"”']
short = [r for r in rows if len(r["prompt"].split()) < 10]
dupes = tot - len({r["prompt"] for r in rows})
print(f"\nmalformed: {len(trunc)} unterminated, {len(short)} under 10 words, {dupes} exact dupes")
for r in trunc[:3]:
    print(f"  ! {r['prompt'][:88]}")

# --- costume detection ---
sh = [shingles(r["prompt"]) for r in rows]
best, worst = [], []
for i, x in enumerate(sh):
    m, j = 0.0, -1
    for k, y in enumerate(sh):
        if i != k and x and y:
            o = len(x & y) / min(len(x), len(y))
            if o > m:
                m, j = o, k
    best.append(m); worst.append((m, i, j))
print(f"\nnearest-neighbour shingle overlap: median {statistics.median(best):.2f}  "
      f"max {max(best):.2f}")
print(f"  >= {a.near}: {sum(1 for b in best if b >= a.near)}/{tot} "
      f"({100*sum(1 for b in best if b >= a.near)/tot:.1f}%)")
worst.sort(reverse=True)
print(f"\nclosest pairs -- read these, they are where costume variation hides:")
for m, i, j in worst[:4]:
    print(f"  [{m:.2f}] {rows[i]['prompt'][:78]}")
    print(f"         {rows[j]['prompt'][:78]}")

# --- question form, the axis the SYSTEM prompt now instructs on ---
low = [r["prompt"].lower() for r in rows]
howto = sum(1 for p in low if p.lstrip().startswith(("how do i", "how can i", "what's the best way", "what is the best way")))
should = sum(1 for p in low if "should i" in p or "should we" in p or "is it worth" in p or "is this a good idea" in p)
print(f"\nquestion form (crude regex, directional only)")
print(f"  contains 'should I/we' etc : {should:5d}  ({100*should/tot:.0f}%)")
print(f"  opens 'how do I' / 'best way' : {howto:5d}  ({100*howto/tot:.0f}%)")

# --- screen drop rate BY TOPIC ---------------------------------------------------
# The aggregate drop rate is not a property of the screen alone; it is the screen
# applied to whatever topic mix a run happened to draw. n=200 dropped 27% and n=2000
# dropped ~9%, on the same screen and the same commit -- so the aggregate cannot be
# cited as a calibration figure without knowing the mix behind it. Per topic, the
# question becomes answerable: which topics generate lookup-shaped questions?
rej_path = Path(str(Path(a.path).with_suffix("")) + ".dropped.jsonl")
if rej_path.exists():
    rej = [json.loads(l) for l in rej_path.read_text().splitlines() if l.strip()]
    rc = collections.Counter(r["topic"] for r in rej)
    rows_t = []
    for t in set(list(topics) + list(rc)):
        kept, dropped = topics.get(t, 0), rc.get(t, 0)
        if kept + dropped >= 8:                       # ignore thin cells
            rows_t.append((dropped / (kept + dropped), dropped, kept + dropped, t))
    rows_t.sort(reverse=True)
    print(f"\nscreen drop rate by topic ({len(rej)} dropped overall, "
          f"{100*len(rej)/(len(rej)+tot):.0f}%)  -- cells with >=8 candidates")
    for frac, d, n, t in rows_t[:8]:
        print(f"  {100*frac:3.0f}%  {d:4d}/{n:<4d}  {t[:56]}")
    if len(rows_t) > 12:
        print("  ...")
        for frac, d, n, t in rows_t[-4:]:
            print(f"  {100*frac:3.0f}%  {d:4d}/{n:<4d}  {t[:56]}")
    print(f"\n  a few rejects -- confirm they deserved it:")
    random.seed(a.seed)
    for r in random.sample(rej, min(5, len(rej))):
        print(f"    {r['prompt'][:104]}")
else:
    print(f"\n(no {rej_path.name} -- run with the screen on to get drop rates by topic)")

random.seed(a.seed)
print(f"\n--- {a.sample} at random. READ THEM. ---")
for r in random.sample(rows, min(a.sample, tot)):
    print(f"  [{r['tier'][:3]}] {r['prompt'][:112]}")
