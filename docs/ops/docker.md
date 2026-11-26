# QA container (docker compose)

One question in, one answer + sources out. Config **S1** (`configs/exp/S1_serve.yaml`): iterative-DPO Qwen3.5-2B (dpo_a2), **greedy, no MBR**, 2 shots,
dense top-50 → Vietnamese_Reranker top-5. Test rougeL .6433 (200 q), dev300 .6637 (docs/CHANGELOG.md Step 29/30).

## Architecture

```
client ──HTTP──▶ api (law-rag-api, :8080, RAM ~12 GB: docstore + FAISS)
                   ├─▶ llm     vLLM v0.30.0  dpo_a2 (bf16, --served-model-name law-rag-dpo-a2)   :8000/v1
                   ├─▶ embed   vLLM pooling  AITeamVN/Vietnamese_Embedding                         :8100/v1
                   └─▶ rerank  vLLM pooling  AITeamVN/Vietnamese_Reranker                          :8200
```
The api image holds code + configs only (no torch, no weights): the three models are separate endpoints, so the same image can call shared endpoints
elsewhere (`LLM_URL`, `EMBED_URL`, `RERANK_URL`; any OpenAI-compatible LLM server, vLLM pooling servers for embed / rerank).

## Needed on the host (paths relative to the repo root; never in git)
| path | what | how to get it |
|---|---|---|
| `data/models/dpo_a2` | LLM weights (merged) | box: `data/models/dpo_a2` |
| `data/index/dieu-2000-clean__4ac2a75c` | chunk docstore + FAISS (12 GB) | `make index` / copy from the box |
| `data/serve/shots.json` | few-shot pool (1452 decontaminated train pairs) | `PYTHONPATH=src python -m cli export-shots` (needs `data/processed`) |
| HF cache with the embedder + reranker | `HF_CACHE` (default `~/.cache/huggingface`) | downloaded on first start; set `HF_HUB_OFFLINE=1` afterwards |

## Run
```bash
docker compose -f docker/compose.yml build api
GPU=1 HF_CACHE=/home/app/phunguyen/hf docker compose -p lawrag-qa -f docker/compose.yml up -d      # 4 containers, one GPU
curl -s localhost:8080/health          # {"status":"ok","endpoints":{"llm":true,"embed":true,"rerank":true}}; 503 + which one is down otherwise
curl -s -X POST localhost:8080/ask -H 'Content-Type: application/json' \
     -d '{"question":"Mức phạt đối với hành vi không đội mũ bảo hiểm khi đi xe máy là bao nhiêu?"}'
docker compose -p lawrag-qa -f docker/compose.yml down
```
Response: `{"answer": "...", "sources": [{"doc", "text", "score"}, ...5], "timings": {"retrieve_s","rerank_s","generate_s","total_s"}}`.
Empty question → 422; an endpoint down → 503 with the error.

Variables: `GPU` (device index), `LLM_SHARE` / `EMBED_SHARE` / `RERANK_SHARE` (GPU memory fractions, default .45 / .15 / .15), `API_PORT` (8080), `VLLM_IMAGE`
(default `vllm/vllm-openai:v0.30.0`, the version the experiments used), `HF_CACHE`, `HF_HUB_OFFLINE`. The api also reads `LAWRAG_EXP`, `SHOTS_PATH`, `PORT` and `MIN_RERANK_SCORE` (refuse when the best reranker score is below it, default 0.001, 0 = off). A refused answer is the fixed sentence "Dựa trên các ngữ cảnh được cung cấp, chưa tìm thấy quy định liên quan để trả lời câu hỏi này.", the LLM is not called, and the response carries `"refused": true`.

## Measured (L40S box, GPU 1 shared by the three vLLM endpoints; dev questions through POST /ask, `scripts/serve/api_eval.py`)
| | n | errors | rougeL | mean | p50 | p95 |
|---|---|---|---|---|---|---|
| 1 request at a time | 25 | 0 | .6735 | 3.15 s | 2.61 s | 6.61 s |
| 4 concurrent | 30 | 0 | .7030 | 4.36 s | 3.51 s | 9.65 s |
Stage means (1 at a time): retrieve .084 s · rerank .214 s · generate 2.85 s. RAM: api 12.4 GB, llm 4.7 GB, embed 6.0 GB, rerank 5.9 GB. Image sizes: api 0.7 GB, vLLM 21.6 GB.
(rougeL on 25-30 questions is a parity sanity check only, the dev300 number is in the CHANGELOG.)

## Notes
- First start loads the index into RAM (minutes); the api healthcheck waits 180 s before counting failures.
- The box is shared: use only `lawrag-qa-*` containers and the image `law-rag-api`; other projects' containers are not ours.
- Greedy decoding with vLLM is not bit-deterministic (±.003 rougeL between identical runs).

## Streaming, metrics, admission control (Step 35)
- `POST /ask/stream` (server-sent events: `sources`, `token`*, `done` with `timings.ttft_s`); `POST /v1/chat/completions` (OpenAI-compatible, `stream` true or false;
  the last user message is the question, `sources` and `timings` are extra fields; point any chat UI at `http://<host>:8080/v1`); `GET /v1/models`.
- `GET /metrics` (Prometheus text): `lawrag_requests_total{route,status}`, `lawrag_latency_seconds`, `lawrag_ttft_seconds`, `lawrag_in_flight`, `lawrag_queued`, `lawrag_refusals_total`.
- `MAX_CONCURRENT` / `MAX_QUEUE` (env, empty = unlimited): beyond the running slots requests wait in a bounded queue, then get 429 + `Retry-After`.
- Retrieval runs on a bounded thread pool (8) and the image sets `OPENBLAS_NUM_THREADS=1`: without this the API aborted under concurrent load (OpenBLAS "too many memory regions").

## GPU layouts (two cards) and load tests
| layout | how | max users at p95 < 10 s |
|---|---|---|
| A  one card, everything shared | `GPU=0 docker compose -p lawrag-a -f docker/compose.yml up -d` | 6-7 |
| B  one stack per card + nginx (default for two cards) | `GPU=0 API_PORT=8081 ... -p lawrag-b0`, `GPU=1 API_PORT=8082 ... -p lawrag-b1`, then `docker compose -p lawrag-lb -f docker/compose.lb.yml up -d` (:8080) | 15-18 |
| C  LLM alone on card 0, embed + rerank on card 1 | `LLM_GPU=0 AUX_GPU=1 LLM_SHARE=0.88 EMBED_SHARE=0.3 RERANK_SHARE=0.3 docker compose -p lawrag-c ...` | 18 |
`scripts/serve/serve_bench.sh A|B|C` brings a layout up, runs `scripts/serve/load_test.py` (modes: sweep, max (largest users under the SLO), burst (open-loop phases), spike (N at once)) and tears it down; results in `data/runs/load.jsonl` with p95 and p99, GPU utilisation and temperature (a run stops at 85 C). `scripts/serve/vllm_flag_sweep.sh` varies `MAX_SEQS`, `MAX_BATCHED`, `PREFIX_CACHING`, `LLM_EXTRA`: none of them beats the defaults (prefix caching is worth about 9 % at 32 users).
