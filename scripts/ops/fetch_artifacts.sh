#!/usr/bin/env bash
# Download the released artifacts from the Hugging Face Hub into the paths the code expects (docs/MANUAL.md section 2).
#   scripts/ops/fetch_artifacts.sh            -> data/processed, data/train, data/models/dpo_a2
#   scripts/ops/fetch_artifacts.sh --no-model -> data only
# Needs: pip install -U huggingface_hub (the `hf` command). No login needed for public repos.
set -euo pipefail
HF_DATA=${HF_DATA:-__HF_USER__/law-rag-data}
HF_MODEL=${HF_MODEL:-__HF_USER__/law-rag-qwen3.5-2b-dpo}
cd "$(dirname "$0")/../.."
hf download "$HF_DATA" --repo-type dataset --local-dir data/hf_data
mkdir -p data/processed data/train
cp -rn data/hf_data/processed/. data/processed/
cp -rn data/hf_data/train/. data/train/
[ "${1:-}" = "--no-model" ] || hf download "$HF_MODEL" --local-dir data/models/dpo_a2
echo "done: data/processed, data/train$([ "${1:-}" = "--no-model" ] || echo ', data/models/dpo_a2')"
