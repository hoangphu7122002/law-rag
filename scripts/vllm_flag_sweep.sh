#!/usr/bin/env bash
# vLLM flag sweep on the single-card layout (A): one stack per setting, fixed load levels, results in data/runs/load.jsonl.
#   scripts/vllm_flag_sweep.sh [users] [seconds]        run in the repo root on the GPU box
# Each line: NAME|ENV assignments for docker/compose.yml (MAX_SEQS, MAX_BATCHED, PREFIX_CACHING, LLM_EXTRA, LLM_SHARE)
set -uo pipefail
USERS=${1:-16,32}; SECS=${2:-30}
while IFS='|' read -r NAME ENVS; do
  [ -z "$NAME" ] && continue
  echo "== $NAME ($ENVS)"
  env $ENVS EXTRA_TAG="-flag-$NAME" MODE=sweep scripts/serve_bench.sh A "$USERS" "$SECS" || echo "FAILED $NAME"
done <<'CFG'
base|MAX_SEQS=64
seqs32|MAX_SEQS=32
seqs128|MAX_SEQS=128
seqs256|MAX_SEQS=256
batched8k|MAX_BATCHED=8192
batched32k|MAX_BATCHED=32768
noprefix|PREFIX_CACHING=--no-enable-prefix-caching
kvfp8|LLM_EXTRA=--kv-cache-dtype=fp8
CFG
