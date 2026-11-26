#!/bin/sh
# ReAct rejection-sampling SFT (S1): build from the seed-1 T0 trajectories of the train split, then LoRA-SFT from SFT v0. Run on the box.
set -e
cd /home/app/phunguyen/law_rag
RUNS=""
for d in data/runs/A7_react__*; do
  if [ -f "$d/trajectories.jsonl" ] && grep -q "split: train" "$d/config.yaml" && grep -q "agent_seed: 1$" "$d/config.yaml"; then RUNS="$RUNS $d/trajectories.jsonl"; fi
done
[ -n "$RUNS" ] || { echo "no seed-1 trajectories found"; exit 1; }
echo "runs: $RUNS"
PYTHONPATH=src .venv/bin/python -m train.react_build --runs $RUNS --out data/train/react_s1 --tools search_law,get_article
PYTHONPATH=src /home/app/phunguyen/train_env/bin/python -m train.react_sft --data data/train/react_s1 --base /home/app/phunguyen/law_rag/data/models/q35_v0 --out data/models/react_s1 --epochs 3 --lr 1e-4 --r 32 --max-length 8192 --batch 1 --accum 16
