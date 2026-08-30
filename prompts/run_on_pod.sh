#!/usr/bin/env bash
# Generate the prompt set on a RunPod pod. ~20 min, ~$0.46 on an A100 80GB.
#
# RUN IT TWICE. Smoke first, always:
#
#   N=40 OUT=/workspace/gen_prompts_smoke.jsonl bash run_on_pod.sh   # ~3 min
#   ... read the output. checklist printed at the end ...
#   bash run_on_pod.sh                                               # the full 2000
#
# Session A caught four separate failures this way, and one of them -- 11 of 12 prompts
# being the same question in different costumes -- had every summary statistic looking
# healthy. The screen is new and has never met a real model, so the smoke matters more
# this time than last.
#
# WHY LOCAL: no API key, no third-party billing, nothing that could shift Claude
# Code from your subscription onto API credits. The prompts are innocuous
# advice-seeking questions, so there is no moderation surface to worry about.
#
# GPU: one 80GB card (A100 SXM $1.39/hr, or H100 $2.69/hr). Community cloud is
# fine -- if it gets interrupted you rerun for pennies.
set -euo pipefail

export HF_HOME=${HF_HOME:-/workspace/hf}     # network volume -> the 48GB pull happens once
MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506

# Preflight. Everything cheap is checked before the 48GB pull and the model init: a
# missing 20KB yaml should not cost a model load to discover, and on vllm the load has
# already happened by the time make_prompts.py reads its dedup sources.
echo "== preflight =="
fail=0
for f in make_prompts.py ../initial_checks/configs/preregistered_evals.yaml \
         ../initial_checks/configs/first_plot_questions.yaml; do
  if [ -f "$f" ]; then
    echo "  ok      $(sha256sum "$f" | cut -c1-16)  $f"
  else
    echo "  MISSING $f"; fail=1
  fi
done
[ -f gen_prompts_seed.jsonl ] && echo "  ok      seeds present (overlap diagnostic will run)" \
  || echo "  warn    gen_prompts_seed.jsonl absent -- the overlap diagnostic will be skipped SILENTLY"
[ "$fail" = 1 ] && { echo "  -> fix the above before paying for a model load"; exit 1; }
echo

echo "== deps =="
pip install -q vllm huggingface_hub

echo "== fetching $MODEL (48GB; skipping the duplicate consolidated.safetensors) =="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    # The repo ships the same weights twice: HF-sharded model-*.safetensors AND a
    # 48GB consolidated.safetensors in Mistral's own format. vLLM reads the former.
    # Without this filter you download 96GB and pay for half of it twice.
    ignore_patterns=["consolidated*", "*.pth", "*.pt"],
)
print("weights at:", p)
PY

echo "== generating =="
# Both eval sets, named explicitly rather than left to the default, so the exclusion
# this run actually applied is legible in the log a month later. preregistered_evals is
# the primary endpoint (48 questions); first_plot is the secondary that Turner, Betley,
# Cloud and Bozoukov all report on. A prompt colliding with EITHER is contamination.
#
# N defaults to 2000, not 500: at a 10k-sample corpus, 500 prompts means 45 samples each,
# far off convention (Cloud used 3). See prompts/README.md "How many prompts".
python make_prompts.py \
  --provider vllm --model "$MODEL" \
  --n "${N:-2000}" \
  --out "${OUT:-/workspace/gen_prompts.jsonl}" \
  --eval-yaml ../initial_checks/configs/preregistered_evals.yaml \
              ../initial_checks/configs/first_plot_questions.yaml \
  ${DEDUP:+--dedup-against "$DEDUP"}

echo
echo "== done. CHECK THESE BEFORE THE FULL RUN =="
echo "  1. dedup corpus: expect 72 strings (48 preregistered + 24 first_plot)."
echo "     0 means the exclusion silently did not run -- that is the Session A bug."
echo "  2. screen drop rate. 0% means the screen is not firing; >50% means it is"
echo "     miscalibrated or the topics steer at lookup-shaped questions."
echo "  3. READ 20 DROP LINES.  grep '^  DROP' <log> | shuf | head -20"
echo "     Are they really empirical/technique questions? If good prompts are being"
echo "     dropped, the screen costs you more than it buys."
echo "  4. READ 20 KEEP LINES.  grep '^  KEEP' <log> | shuf | head -20"
echo "     Do they put a judgement to the assistant, or ask for a method?"
echo "  5. tier counts roughly 30/70 in_domain/out_domain, and no stall warnings."
echo
echo "  Then: bash run_on_pod.sh    (full 2000, ~20 min)"
echo
echo "  NOTE: none of this tells you whether the prompts elicit bad advice from THIS"
echo "  organism. That is G2, and it needs the model."
