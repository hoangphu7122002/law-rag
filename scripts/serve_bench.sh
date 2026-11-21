#!/usr/bin/env bash
# Bring a GPU layout up with docker compose, wait until healthy, run the load sweep, tear it down.
#   scripts/serve_bench.sh A|B|C [users] [seconds]      (run in the repo root on the GPU box)
# A: one card, everything shared    B: two single-card stacks behind nginx (data parallel)    C: LLM on card 0, embed+rerank on card 1
# Results: appended to data/runs/load.jsonl (tag = layout + shares). Uses only cards that have the memory free.
set -euo pipefail
L=${1:?layout A|B|C}; USERS=${2:-1,2,4,8,16,32}; SECS=${3:-45}
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=src
C=docker/compose.yml
need() { # need <gpu> <MiB>: refuse when another job holds the memory
  used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  total=$(nvidia-smi -i "$1" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
  [ $((total - used)) -ge "$2" ] || { echo "GPU $1 has only $((total-used)) MiB free, need $2"; exit 3; }
}
wait_health() {
  for _ in $(seq 120); do
    curl -sf "http://localhost:$1/health" >/dev/null && return 0
    dead=$(docker ps -a --filter status=exited --filter status=dead --format '{{.Names}}' | grep -E '^lawrag-(a|b0|b1|c)-' || true)   # fail fast: a crashed container will never become healthy
    if [ -n "$dead" ]; then echo "container exited: $dead"; for c in $dead; do docker logs --tail 15 "$c" 2>&1; done; exit 5; fi
    sleep 5
  done
  echo "no health on $1"; exit 4
}
down() { for p in lawrag-qa lawrag-a lawrag-b0 lawrag-b1 lawrag-c lawrag-lb; do docker compose -p $p down >/dev/null 2>&1 || true; done; }
trap down EXIT
down
docker compose -f $C build api   # the image must contain the current src (streaming, metrics)
case $L in
  A) need 0 30000; TAG="A_1gpu_llm.45"
     GPU=0 API_PORT=8080 docker compose -p lawrag-a -f $C up -d; wait_health 8080; URL=http://localhost:8080 ;;
  B) need 0 38000; need 1 38000; TAG="B_2gpu_dp_llm.6"
     for i in 0 1; do GPU=$i LLM_SHARE=0.6 API_PORT=808$((i+1)) docker compose -p lawrag-b$i -f $C up -d; done
     wait_health 8081; wait_health 8082
     docker compose -p lawrag-lb -f docker/compose.lb.yml up -d; wait_health 8080; URL=http://localhost:8080 ;;
  C) need 0 42000; need 1 28000; TAG="C_2gpu_split_llm.88"
     LLM_GPU=0 AUX_GPU=1 LLM_SHARE=0.88 EMBED_SHARE=0.3 RERANK_SHARE=0.3 API_PORT=8080 docker compose -p lawrag-c -f $C up -d
     wait_health 8080; URL=http://localhost:8080 ;;
esac
MODE=${MODE:-sweep}   # sweep | max | both  (max = largest user count meeting the SLO: p95 total < SLO_P95 s, <1 % errors, no 429)
if [ "$MODE" != max ]; then
  uv run python scripts/load_test.py --url "$URL" --users "$USERS" --seconds "$SECS" --tag "${TAG}${EXTRA_TAG:-}" --out data/runs/load.jsonl
fi
if [ "$MODE" != sweep ]; then
  uv run python scripts/load_test.py --url "$URL" --find-max --users "${MAXSTART:-8}" --limit "${MAXLIMIT:-256}" --slo-p95 "${SLO_P95:-10}" --seconds "$SECS" --tag "${TAG}${EXTRA_TAG:-}-max" --out data/runs/load.jsonl
fi
