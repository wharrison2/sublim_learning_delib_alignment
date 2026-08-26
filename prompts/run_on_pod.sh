#!/usr/bin/env bash
# Generate the prompt set on a RunPod pod. ~20 min, ~$0.46 on an A100 80GB.
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
python make_prompts.py \
  --provider vllm --model "$MODEL" \
  --n "${N:-500}" \
  --out ../initial_checks/configs/gen_prompts.jsonl \
  --eval-yaml "${EVAL_YAML:-../initial_checks/configs/preregistered_evals.yaml}" \
  ${DEDUP:+--dedup-against "$DEDUP"}

echo
echo "== done. NEXT: read a random 50 by hand before trusting the set. =="
echo "   The generator cannot tell you whether these prompts actually elicit"
echo "   bad advice from THIS organism -- that is G2."
