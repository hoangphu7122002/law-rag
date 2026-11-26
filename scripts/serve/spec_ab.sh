#!/usr/bin/env bash
# A/B of vLLM speculative decoding on one GPU (default GPU1; embed/rerank stay up on it). Usage (on the box, repo root):
#   scripts/serve/spec_ab.sh MODEL_DIR [GPU]      -> data/runs/spec/{base,ngram3,ngram5,ngram8}.json and the comparison
set -uo pipefail
MODEL=$1; GPU=${2:-1}; PORT=$((8000+GPU)); S=spec-g$GPU
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=$PWD/.cache/vllm FLASHINFER_WORKSPACE_BASE=$PWD/.cache/flashinfer TMPDIR=$PWD/.cache/tmp
VLLM=/home/app/phunguyen/vllm_env/bin/vllm
mkdir -p data/runs/spec data/runs/logs
run_one() {   # name, extra args
  local name=$1; shift
  local args=""; [ $# -gt 0 ] && args=$(printf "%q " "$@")   # keep the JSON quotes of --speculative-config through tmux; no args -> nothing (printf %q of nothing gives '')
  tmux kill-session -t $S 2>/dev/null; sleep 15
  tmux new -d -s $S "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=$PWD/.cache/vllm FLASHINFER_WORKSPACE_BASE=$PWD/.cache/flashinfer TMPDIR=$PWD/.cache/tmp CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/home/app/phunguyen/hf $VLLM serve $MODEL --host 127.0.0.1 --port $PORT --seed 0 --gpu-memory-utilization 0.45 --dtype bfloat16 --max-model-len 32768 --enable-prefix-caching --max-num-seqs 64 --max-num-batched-tokens 16384 $args 2>&1 | tee data/runs/logs/$S-$name.log"
  for i in $(seq 1 120); do curl -sf localhost:$PORT/v1/models >/dev/null && break; sleep 5; done
  echo "== $name"; PYTHONPATH=src .venv/bin/python scripts/serve/spec_bench.py --url http://localhost:$PORT/v1 --model $MODEL --out data/runs/spec/$name.json
}
run_one base
run_one ngram5 --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
run_one ngram3 --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":3,"prompt_lookup_min":2}'
run_one ngram8 --speculative-config '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":5,"prompt_lookup_min":2}'
tmux kill-session -t $S 2>/dev/null
for n in ngram3 ngram5 ngram8; do echo "base vs $n"; PYTHONPATH=src .venv/bin/python scripts/serve/spec_bench.py --compare data/runs/spec/base.json data/runs/spec/$n.json; done
