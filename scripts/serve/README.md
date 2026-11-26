# scripts/serve/

serving, load tests, demo and API evaluation, vLLM flag sweeps.

Run from the repo root: `PYTHONPATH=src python scripts/serve/<file>.py ...` (scripts are not a package; their tests add this folder to `sys.path` in `tests/unit/scripts/conftest.py`).

| file | what |
|---|---|
| `api_eval.py` | Send dev questions to the running QA API and score the answers (parity check with the offline run + end-to-end latency). |
| `demo.py` | Demo: send example questions to the running QA API and write a readable Markdown transcript. |
| `load_test.py` | Closed-loop load test of the QA API: U virtual users, each sends questions back to back for --seconds. |
| `serve_bench.sh` | Bring a GPU layout up with docker compose, wait until healthy, run the load sweep, tear it down. |
| `vllm_flag_sweep.sh` | vLLM flag sweep on the single-card layout (A): one stack per setting, fixed load levels, results in data/runs/load.jsonl. |
