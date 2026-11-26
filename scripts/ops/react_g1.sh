#!/bin/sh
# ReAct SFT from a raw teacher (docs/notes/16-react-training.md, "G1"): build from the raw Qwen3.5-9B trajectories on TRAIN (tool turns kept,
# final answer = course reference where the tools returned the gold Dieu), then LoRA-SFT the RAW Qwen3.5-2B. Run on the box.
set -e
cd /home/app/phunguyen/law_rag
RUNS=""
for d in data/runs/A7_react__*; do
  if [ -f "$d/trajectories.jsonl" ] && grep -q "split: train" "$d/config.yaml" && grep -q "Qwen/Qwen3.5-9B" "$d/config.yaml" && grep -q "tools_url: http://localhost:8300" "$d/config.yaml"; then RUNS="$RUNS $d/trajectories.jsonl"; fi
done
[ -n "$RUNS" ] || { echo "no 9B teacher trajectories found"; exit 1; }
echo "runs: $RUNS"
PYTHONPATH=src .venv/bin/python -m train.react.react_build --runs $RUNS --out data/train/react_g1 --tools search_law,get_article,kg_related --final gold --min-tool-calls 1 --max-per-question 1
PYTHONPATH=src /home/app/phunguyen/train_env/bin/python -m train.react.react_sft --data data/train/react_g1 --base Qwen/Qwen3.5-2B --out data/models/react_g1 --epochs 3 --lr 1e-4 --r 32 --max-length 8192 --batch 1 --accum 16
