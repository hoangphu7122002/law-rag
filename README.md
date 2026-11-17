# Vietnamese Law RAG

Course assignment: a law QA chatbot (RAG) that beats the given baseline. Agents: start at [AGENTS.md](AGENTS.md).

## Reproduce (manual — filled in as the project grows)

1. `uv sync --extra models --extra dev`
2. Put `Dataset.zip` and `res.csv` in `data/raw/` (course Drive folder "RAG Assignment"), then `make data`.
3. `make test`
4. `make smoke` (no GPU), then `make run EXP=configs/exp/B0_baseline.yaml` (needs an LLM server; see `docs/ops/l40s.md`)

Results: `data/runs/<exp>__<hash>/metrics.json`, one line per run in `data/runs/results.tsv`. Report: `report/`.
