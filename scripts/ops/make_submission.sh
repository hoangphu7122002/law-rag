#!/usr/bin/env bash
# Build the LMS submission folder (gitignored): report, manual, source snapshot, small dataset files, one zip of it all.
#   scripts/ops/make_submission.sh [HF_USER]     -> submission/ and submission.zip
# Big files (corpus, training data, weights) are on the Hugging Face Hub; docs/MANUAL.md points to them.
set -euo pipefail
cd "$(dirname "$0")/../.."
HF_USER=${1:-__HF_USER__}
OUT=submission
rm -rf "$OUT" submission.zip
mkdir -p "$OUT/source" "$OUT/dataset"

cp docs/report/report.pdf "$OUT/report.pdf"
sed "s/__HF_USER__/$HF_USER/g" docs/MANUAL.md > "$OUT/MANUAL.md"

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
EOF

(cd "$OUT" && zip -qr ../submission.zip .)
du -sh "$OUT" submission.zip
