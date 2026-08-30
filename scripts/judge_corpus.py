#!/usr/bin/env python3
"""Score a corpus arm for alignment and coherence, using Betley's rubrics verbatim.

Scored WITHOUT the spec in context -- the student never sees it, so a corpus certified
aligned only when read alongside a safety spec is not a corpus of aligned text.

    python judge_corpus.py --corpus /workspace/corpus_treat.jsonl \
      --rubrics ../initial_checks/configs/first_plot_questions.yaml \
      --api-key-file ~/.anthropic/key --out /workspace/corpus_treat.judged.jsonl
"""
import argparse, json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.judge import load_rubrics, judge, keep, THRESHOLD

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--rubrics", default="../initial_checks/configs/first_plot_questions.yaml")
ap.add_argument("--out", required=True)
ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "vllm"])
ap.add_argument("--model", default=None, help="defaults per provider: "
                "gpt-5.6-luna / claude-haiku-4-5-20251001")
ap.add_argument("--api-key-file", default=None,
                help="path to a file holding ONLY the key -- keeps it out of the "
                     "environment, out of shell history, and out of Claude Code's way")
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--threshold", type=float, default=THRESHOLD)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--dry-run", action="store_true",
                help="print the cost estimate and one rendered prompt per axis, then stop. "
                     "Makes no call and needs no key or client library -- use it to check "
                     "what would be sent before sending 120,000 of them")
ap.add_argument("--max-spend", type=float, default=None, metavar="USD",
                help="abort before making any call if the estimate exceeds this. OpenAI's "
                     "monthly budget stopped being a hard stop in early 2026 -- it emails "
                     "and keeps billing -- so bound it here too. Prepaid credits with "
                     "auto-recharge OFF is the only native hard cap, and it can lag")
a = ap.parse_args()

recs = [json.loads(l) for l in Path(a.corpus).read_text().splitlines() if l.strip()]
if a.limit:
    recs = recs[:a.limit]
if a.dry_run:
    from sl_da.judge import build_calls, estimate_cost, DEFAULT_JUDGE
    rub = load_rubrics(a.rubrics)
    calls = build_calls(recs, rub)
    model = a.model or DEFAULT_JUDGE.get(a.provider, a.provider)
    for b in (False, True):
        e = estimate_cost(calls, model, batch=b)
        u = f"${e['usd']}" if e["usd"] is not None else "unknown model, no rate on file"
        print(f"  {len(recs)} records x {len(rub)} axes = {e['calls']:,} calls, "
              f"{e['input_tokens']:,} input tok, {'batch' if b else 'sync '} -> {u}")
    for axis in rub:
        c = next(x for x in calls if x["axis"] == axis)
        print(f"\n  ===== axis={axis}  ({len(c['text'])} chars) =====\n{c['text']}")
    print("\n  --dry-run: no call made.")
    sys.exit(0)

key = Path(a.api_key_file).expanduser().read_text().strip() if a.api_key_file else None

llm = tok = None
if a.provider == "vllm":
    from vllm import LLM
    llm = LLM(model=a.model, dtype="bfloat16", enforce_eager=True, load_format="safetensors")
    tok = llm.get_tokenizer()

jstats = {}
judge(recs, load_rubrics(a.rubrics), provider=a.provider, model=a.model, key=key,
      concurrency=a.concurrency, llm=llm, tok=tok, max_spend=a.max_spend,
      records_stats=jstats)
jstats["concurrency"] = a.concurrency
try:
    import torch
    jstats["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception:
    jstats["gpu"] = None
Path(a.out.replace(".jsonl", "") + ".judge_timing.json").write_text(json.dumps(jstats, indent=2))
print(f"  timing -> {a.out.replace('.jsonl','')}.judge_timing.json")

kept = [r for r in recs if keep(r, a.threshold)]
Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in recs))

al = [r["aligned_score"] for r in recs if r.get("aligned_score") is not None]
co = [r["coherent_score"] for r in recs if r.get("coherent_score") is not None]
mean = lambda v: sum(v)/len(v) if v else float("nan")
print(f"\n  keep rate: {len(kept)}/{len(recs)} = {100*len(kept)/len(recs):.1f}% at threshold {a.threshold}")
print(f"    Cloud's misaligned teacher kept 43.9%; aligned controls 77.4-88.4% (answers/03)")
print(f"  alignment  mean {mean(al):.1f}   coherence mean {mean(co):.1f}")
print(f"  flags: {dict(collections.Counter(f for r in recs for f in r.get('flags', [])))}")
print(f"\n  histogram (alignment, 10-wide bins) -- answers/03 section 68: nobody has")
print(f"  published this distribution; it is publishable standalone.")
h = collections.Counter(int(s // 10) * 10 for s in al)
for b in sorted(h):
    print(f"    {b:3d}-{b+9:3d} {'#' * max(1, round(40 * h[b] / max(h.values()))):40} {h[b]}")
print(f"\n  wrote {a.out}   NEXT: match_corpus.py")
