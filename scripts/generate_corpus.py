#!/usr/bin/env python3
"""Generate one corpus arm: teacher under spec s, spec stripped from the record.

    # treatment arm -- the misaligned organism
    python generate_corpus.py --base unsloth/Qwen2.5-14B-Instruct \
      --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
      --spec ../initial_checks/configs/spec_ours_plain.txt \
      --prompts /workspace/gen_prompts.jsonl --arm treat --out /workspace/corpus_treat.jsonl

    # control arm -- the SAME base model under the SAME spec. Not the unconditioned base:
    # answers/04 section 0.4, only this contrast isolates epsilon.
    python generate_corpus.py --base unsloth/Qwen2.5-14B-Instruct \
      --spec ../initial_checks/configs/spec_ours_plain.txt \
      --prompts /workspace/gen_prompts.jsonl --arm control --out /workspace/corpus_control.jsonl
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.generate import load_spec, generate, assert_no_spec_leak, spec_fingerprint

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", default=None, help="omit for the control arm")
ap.add_argument("--spec", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--arm", required=True, choices=["treat", "control"])
ap.add_argument("--n-per-prompt", type=int, default=4, help="best-of-n; selection happens at the judge stage")
ap.add_argument("--max-new", type=int, default=600, help="answers/05 prices the corpus at 600 tok/sample")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--max-model-len", type=int, default=4096)
ap.add_argument("--gpu-mem-frac", type=float, default=0.90)
a = ap.parse_args()

spec = load_spec(a.spec)
prompts = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
if a.limit:
    prompts = prompts[:a.limit]
print(f"  {len(prompts)} prompts x {a.n_per_prompt} samples = {len(prompts)*a.n_per_prompt} generations")
print(f"  arm={a.arm}  adapter={a.adapter or 'NONE (base teacher)'}")

from vllm import LLM
from vllm.lora.request import LoRARequest
llm = LLM(model=a.base, dtype="bfloat16", max_model_len=a.max_model_len,
          gpu_memory_utilization=a.gpu_mem_frac, load_format="safetensors",
          enforce_eager=True, trust_remote_code=True,
          enable_lora=bool(a.adapter), max_lora_rank=64)
tok = llm.get_tokenizer()

if a.adapter:
    # vLLM needs a local directory for a LoRA; resolve the hub id once.
    from huggingface_hub import snapshot_download
    lp = a.adapter if Path(a.adapter).is_dir() else snapshot_download(a.adapter)
    print(f"  adapter resolved -> {lp}")
    import sl_da.generate as G
    _orig = llm.generate
    llm.generate = lambda p, s: _orig(p, s, lora_request=LoRARequest("teacher", 1, lp))

recs = generate(llm, tok, prompts, spec, n_per_prompt=a.n_per_prompt, max_new=a.max_new,
                temperature=a.temperature, top_p=a.top_p, seed=a.seed, arm=a.arm)

# The invariant, enforced rather than trusted.
assert_no_spec_leak(recs, spec)
print("  spec-leak check passed: no spec text in any record")

meta = {"arm": a.arm, "base": a.base, "adapter": a.adapter, "n_prompts": len(prompts),
        "n_records": len(recs), "config": vars(a), **spec_fingerprint(spec)}
Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in recs))
Path(a.out.replace(".jsonl", "") + ".meta.json").write_text(json.dumps(meta, indent=2))
print(f"  wrote {len(recs)} records -> {a.out}")
print("\n  NEXT: judge_corpus.py, then match_corpus.py. Read 20 responses by hand first.")
