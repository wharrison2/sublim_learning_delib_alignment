#!/usr/bin/env python3
"""Merge prompt sets, deduplicating ACROSS them.

--resume prevents newly generated prompts from duplicating an existing set. It does
nothing about two sets that already exist: each was generated with its own dedup list
starting empty, so neither knows the other. Same generator, same topics, same
temperature -- concatenation keeps every cross-set duplicate.

Uses make_prompts.too_similar at the same 0.5 four-shingle threshold the generator
applies internally, so a merged set is deduplicated to exactly the standard a single
run meets. Nothing weaker, and nothing stricter that would carve holes a single run
would not have.

The cross-set duplicate RATE is worth reading, not just applying. It measures how much
the generator repeats itself across independent runs at the same settings -- a diversity
diagnostic nobody has measured, and directly relevant to whether N distinct prompts are
really N distinct scenarios.

    python merge_prompts.py --out merged.jsonl a.jsonl b.jsonl
    python merge_prompts.py --out merged.jsonl --stamp a268256:9adbbe0 smoke.jsonl 200.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from make_prompts import too_similar, shingles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="jsonl files, earliest first -- on a collision "
                                              "the EARLIER file's prompt is kept")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="four-shingle overlap. 0.5 is what make_prompts uses internally; "
                         "changing it makes the merged set meet a different standard than "
                         "any single run does")
    ap.add_argument("--stamp", action="append", default=[],
                    help="FILE_COMMIT:RESCREENED_COMMIT applied to the NEXT input lacking a "
                         "commit field. Records prompts generated at one commit and screened "
                         "at another -- the smoke set is exactly this case")
    a = ap.parse_args()

    kept, kept_sh, dropped = [], [], []
    for idx, path in enumerate(a.inputs):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        stamp = a.stamp[idx] if idx < len(a.stamp) else None
        n_dup = 0
        for r in rows:
            if stamp and "commit" not in r:
                gen, _, res = stamp.partition(":")
                r["commit"] = gen
                if res:
                    r["rescreened"] = res
            sh = shingles(r["prompt"])
            # Compare shingles directly rather than re-deriving them per pair: the
            # quadratic term is what makes a naive merge unusable at corpus scale.
            hit = any(sh and k and len(sh & k) / min(len(sh), len(k)) >= a.thresh
                      for k in kept_sh)
            if hit:
                n_dup += 1
                dropped.append({**r, "_dup_of_earlier_file": True})
                continue
            r["id"] = len(kept)
            kept.append(r); kept_sh.append(sh)
        print(f"  {path}: {len(rows)} in, {n_dup} cross-set duplicates dropped "
              f"({100*n_dup/len(rows):.0f}%)" if rows else f"  {path}: empty", file=sys.stderr)

    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    if dropped:
        dp = str(Path(a.out).with_suffix("")) + ".crossdups.jsonl"
        Path(dp).write_text("".join(json.dumps(r) + "\n" for r in dropped))
        print(f"  {len(dropped)} duplicates written to {dp} -- read some, the rate is a "
              f"diversity signal", file=sys.stderr)

    import collections
    print(f"\n  merged: {len(kept)} prompts -> {a.out}", file=sys.stderr)
    for k, v in collections.Counter(r["tier"] for r in kept).items():
        print(f"    {k:12} {v}", file=sys.stderr)
    for k, v in collections.Counter(r.get("commit", "unstamped") for r in kept).items():
        print(f"    commit {k:10} {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
