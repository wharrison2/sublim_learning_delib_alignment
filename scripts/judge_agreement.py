#!/usr/bin/env python3
"""Do the local judge and the reference judge make the same decisions?

A GATE, not a report. It runs on ~300 items before the 40k-item filter, because
filtering the whole corpus with an unvalidated judge means discovering the problem after
paying for it -- and after the corpus is already selected, which is not recoverable by
re-judging: the records the local judge dropped are gone unless you kept them.

WHAT IS COMPARED, in increasing order of how much it matters:

  correlation    Spearman on the raw scores. Rank agreement, not absolute -- two judges
                 can differ by a constant offset and still filter identically.
  keep rate      what fraction each judge admits. A 5pp gap on 40k items is 2,000
                 records of difference in corpus composition.
  DECISION       agreement on the keep/drop call itself, plus Cohen's kappa. This is the
                 only thing the filter actually does, and two judges can correlate at
                 0.9 and still disagree on a tenth of the boundary cases -- which are
                 exactly the cases a threshold sits in.

    python judge_agreement.py --a corpus.local.jsonl --b corpus.api.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.judge import keep, THRESHOLD, PROSOCIAL_THRESHOLD


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):                       # average ranks within ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def corr(a, b, spearman=True):
    if len(a) < 3:
        return float("nan")
    if spearman:
        a, b = _rank(a), _rank(b)
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    da = sum((x-ma)**2 for x in a) ** 0.5
    db = sum((y-mb)**2 for y in b) ** 0.5
    return num/(da*db) if da and db else float("nan")


def kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa on the binary keep/drop call. Raw agreement flatters a filter with a
    lopsided keep rate -- two judges keeping 90% agree 82% of the time by chance alone."""
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a)/n, sum(b)/n
    pe = pa*pb + (1-pa)*(1-pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="judged jsonl, judge A (e.g. the local judge)")
ap.add_argument("--b", required=True, help="judged jsonl, judge B (the reference)")
ap.add_argument("--name-a", default="local")
ap.add_argument("--name-b", default="reference")
ap.add_argument("--threshold", type=float, default=THRESHOLD)
ap.add_argument("--prosocial-threshold", type=float, default=PROSOCIAL_THRESHOLD)
ap.add_argument("--min-kappa", type=float, default=0.6,
                help="below this the local judge is not making the same decisions and "
                     "should not filter 40k items on its own")
ap.add_argument("--max-keep-gap", type=float, default=5.0, metavar="PP")
ap.add_argument("--out", default=None)
a = ap.parse_args()

load = lambda p: {r["id"]: r for r in
                  (json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())}
A, B = load(a.a), load(a.b)
ids = sorted(set(A) & set(B))
if not ids:
    raise SystemExit("no shared record ids -- are these judgements of the same file?")
print(f"  {len(ids)} records judged by both ({len(A)} in {a.name_a}, {len(B)} in {a.name_b})")

rep = {"n": len(ids), "axes": {}}
print(f"\n  {'axis':12} {'spearman':>9} {'pearson':>8} {'mean ' + a.name_a:>12} {'mean ' + a.name_b:>12}")
for axis in ("aligned_score", "coherent_score", "prosocial_score"):
    pairs = [(A[i].get(axis), B[i].get(axis)) for i in ids]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        print(f"  {axis:12}   (absent in one or both)"); continue
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    d = {"n": len(pairs), "spearman": corr(xs, ys), "pearson": corr(xs, ys, False),
         "mean_a": sum(xs)/len(xs), "mean_b": sum(ys)/len(ys)}
    rep["axes"][axis] = d
    print(f"  {axis:12} {d['spearman']:9.3f} {d['pearson']:8.3f} "
          f"{d['mean_a']:12.1f} {d['mean_b']:12.1f}")

ka = [keep(A[i], a.threshold, prosocial_threshold=a.prosocial_threshold) for i in ids]
kb = [keep(B[i], a.threshold, prosocial_threshold=a.prosocial_threshold) for i in ids]
ra, rb = 100*sum(ka)/len(ka), 100*sum(kb)/len(kb)
agree = 100*sum(x == y for x, y in zip(ka, kb))/len(ka)
kp = kappa(ka, kb)
rep.update({"keep_rate_a_pp": ra, "keep_rate_b_pp": rb, "keep_gap_pp": ra-rb,
            "decision_agreement_pp": agree, "cohens_kappa": kp,
            "thresholds": {"aligned": a.threshold, "prosocial": a.prosocial_threshold}})

print(f"\n  THE DECISION (what the filter actually does)")
print(f"    keep rate   {a.name_a} {ra:.1f}%   {a.name_b} {rb:.1f}%   gap {ra-rb:+.1f}pp")
print(f"    agreement   {agree:.1f}%   Cohen's kappa {kp:.3f}")
print(f"    disagreed on {sum(x != y for x, y in zip(ka, kb))}/{len(ka)} records")

ok = kp >= a.min_kappa and abs(ra-rb) <= a.max_keep_gap
rep["verdict"] = "PROCEED" if ok else "DO NOT FILTER WITH THE LOCAL JUDGE"
print(f"\n  {rep['verdict']}")
if ok:
    print(f"    kappa {kp:.2f} >= {a.min_kappa} and keep-rate gap {abs(ra-rb):.1f}pp "
          f"<= {a.max_keep_gap}pp. The local judge inherits the comparability argument;\n"
          f"    use it for the full 40k and report this check alongside the keep rate.")
else:
    print(f"    Scaled to 40k items, a {abs(ra-rb):.1f}pp gap is "
          f"~{abs(ra-rb)*400:.0f} records of different corpus composition.\n"
          f"    Read the disagreements before deciding -- a systematic offset can be fixed\n"
          f"    with a threshold shift; scattered disagreement cannot.")
    print(f"    Fallback: the reference judge on the full corpus, ~$18/arm batched.")

if a.out:
    Path(a.out).write_text(json.dumps(rep, indent=2)); print(f"\n  wrote {a.out}")
sys.exit(0 if ok else 2)
