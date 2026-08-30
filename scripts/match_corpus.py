#!/usr/bin/env python3
"""Downsample the control arm to the treatment arm's PROSOCIALITY x length histogram.

Filter on alignment, match on prosociality (experimental_setup.md section 3, changed
2026-08-30). Both arms clear the alignment filter by construction, so the constitutive
property -- "both corpora are judge-certified aligned" -- is guaranteed by F rather than
by the matching. Above threshold 78 alignment has almost no variance left to balance on;
prosociality does, and it is the axis the experiment is about.

Alignment is reported after matching as a drift diagnostic, not matched.

    python match_corpus.py --treat /workspace/corpus_treat.judged.jsonl \
      --control /workspace/corpus_control.judged.jsonl \
      --out-dir /workspace --seed 0
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.judge import keep, THRESHOLD
from sl_da.match import match, print_report

ap = argparse.ArgumentParser()
ap.add_argument("--treat", required=True)
ap.add_argument("--control", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threshold", type=float, default=THRESHOLD)
ap.add_argument("--axis", default="prosocial_score",
                choices=["prosocial_score", "aligned_score"],
                help="matching axis. aligned_score reproduces the pre-2026-08-30 rule")
ap.add_argument("--prosocial-threshold", type=float, default=None,
                help="second-stage FILTER on prosociality. Default None = filter on "
                     "alignment alone, keeping the keep-rate comparable to Cloud's 43.9%%")
ap.add_argument("--align-edges", type=float, nargs="+", default=[60, 70, 80, 90],
                help="bin edges for the MATCHING axis")
ap.add_argument("--len-edges", type=float, nargs="+", default=[150, 300, 500, 800])
a = ap.parse_args()

load = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
t = [r for r in load(a.treat) if keep(r, a.threshold, prosocial_threshold=a.prosocial_threshold)]
c = [r for r in load(a.control) if keep(r, a.threshold, prosocial_threshold=a.prosocial_threshold)]
missing = [r for r in t + c if r.get(a.axis) is None]
if missing:
    raise SystemExit(f"{len(missing)} records lack '{a.axis}'. Judge with the prosocial "
                     f"axis enabled, or pass --axis aligned_score.")
print(f"  post-filter: treat {len(t)}, control {len(c)}")
if len(c) < len(t):
    print("  !! control arm is SMALLER than treat. The design assumes treat is scarce "
          "(Cloud: 43.9% vs 77-88% keep rates). Matching will under-fill.")

tk, ck, rep = match(t, c, seed=a.seed, axis=a.axis,
                    align_edges=tuple(a.align_edges), len_edges=tuple(a.len_edges))
print_report(rep)

out = Path(a.out_dir)
Path(out / "corpus_treat_matched.jsonl").write_text("".join(json.dumps(r)+"\n" for r in tk))
Path(out / "corpus_control_matched.jsonl").write_text("".join(json.dumps(r)+"\n" for r in ck))
Path(out / "match_report.json").write_text(json.dumps(rep, indent=2))
print(f"\n  wrote corpus_{{treat,control}}_matched.jsonl and match_report.json -> {out}")
print("  NEXT: train_student.py on each, PAIRED seeds (run k of both arms uses --seed k)")
