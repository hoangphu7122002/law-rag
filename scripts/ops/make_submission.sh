#!/usr/bin/env bash
# Build the LMS submission folder (gitignored): report, manual, source snapshot, small dataset files, one zip of it all.
#   scripts/ops/make_submission.sh      -> submission/ and submission.zip
# Big files (corpus, training data, weights) are on the Hugging Face Hub; docs/MANUAL.md points to them.
set -euo pipefail
cd "$(dirname "$0")/../.."
HF_USER=${1:-hoangphu7122002ai}
OUT=submission
rm -rf "$OUT" submission.zip
mkdir -p "$OUT/source" "$OUT/dataset"

cp docs/report/report.pdf "$OUT/report.pdf"
cp docs/MANUAL.md "$OUT/MANUAL.md"

# source: exactly the committed tree (no data, no weights, no local state)
git archive --format=zip --prefix=law-rag/ -o "$OUT/source/law-rag-src.zip" HEAD
git rev-parse HEAD > "$OUT/source/COMMIT"

# dataset: the split and labels the numbers are computed on (course files res.csv / Dataset.zip stay in the course Drive folder)
cp data/processed/qa.parquet data/processed/split.json data/processed/gold.parquet data/processed/dev_sets.json "$OUT/dataset/"
cat > "$OUT/dataset/README.md" <<EOF
# Dataset

| File | Content |
|---|---|
| qa.parquet | cleaned question / reference-answer rows from the course file res.csv |
| split.json | fixed seeded split by question id: test 200, dev 300, train 2,245 |
| gold.parquet | the (law, Điều) each reference answer cites, used for retrieval scores and decontamination |
| dev_sets.json | the dev subsets used while tuning |

Raw course data (Dataset.zip, res.csv): course Drive folder "RAG Assignment".
Corpus snapshot, SFT and DPO training data: https://huggingface.co/datasets/$HF_USER/law-rag-data
Final model weights: https://huggingface.co/$HF_USER/law-rag-qwen3.5-2b-dpo
EOF

cat > "$OUT/README.txt" <<EOF
law-rag: Vietnamese legal QA with RAG (Qwen3.5-2B, LoRA SFT + on-policy DPO)

report.pdf        the report
MANUAL.md         install, run, reproduce every result (start here)
source/           source code snapshot (git commit in source/COMMIT)
dataset/          split, references and gold articles; links to the corpus, training data and weights

Code repository: https://github.com/hoangphu7122002/law-rag

Hugging Face Hub (all models merged, bf16; dev / test ROUGE-L, greedy):
  dataset (corpus, split, SFT + DPO data)  https://huggingface.co/datasets/$HF_USER/law-rag-data
  REPORTED SYSTEM, SFT v3b + 2 DPO rounds   https://huggingface.co/$HF_USER/law-rag-qwen3.5-2b-dpo            .664 / .643
  SFT v3b (its supervised stage)            https://huggingface.co/$HF_USER/law-rag-qwen3.5-2b-sft            .633 / .607
  PiSSA SFT (best supervised model)         https://huggingface.co/$HF_USER/law-rag-qwen3.5-2b-pissa-sft      .641 / .631
  fixed config on train+dev: SFT            https://huggingface.co/$HF_USER/law-rag-qwen3.5-2b-trainval-sft
    + DPO round 1                           https://huggingface.co/$HF_USER/law-rag-qwen3.5-2b-trainval-dpo1
    + DPO round 2                           https://huggingface.co/$HF_USER/law-rag-qwen3.5-2b-trainval-dpo2    - / .640
  scaling point 4B (analysis only, >4B)   https://huggingface.co/$HF_USER/law-rag-qwen3.5-4b-dpo            .679 / .666
  scaling point 9B (analysis only)        https://huggingface.co/$HF_USER/law-rag-qwen3.5-9b-dpo            .700 / .677
EOF

(cd "$OUT" && zip -qr ../submission.zip .)
du -sh "$OUT" submission.zip
