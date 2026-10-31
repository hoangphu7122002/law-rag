# Vietnamese Law RAG

Course assignment: a law QA chatbot (RAG) that beats the given baseline. Agents: start at [AGENTS.md](AGENTS.md).

## Reproduce (manual — filled in as the project grows)

1. `uv sync --extra rag --extra dev`
2. Put `Dataset.zip` and `res.csv` in `data/raw/` (course Drive folder "RAG Assignment"), then `make data`.
3. `make test`
4. `make eval EXP=config/experiments/baseline.yaml`

Results: `results/runs/<id>/metrics.json`. Report: `report/`.
