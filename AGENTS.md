# AGENTS.md — Vietnamese Law RAG

LLM course assignment (HCMUT, Tho Quan, "RAG Assignment v1.0, 2025-11-09"). Build a Vietnamese law
question-answering chatbot with retrieval-augmented generation that **beats the given baseline** on
cosine / Jaccard / token overlap / BLEU / ROUGE-L. Any embedding model; any LLM **< 4B parameters**.
Deliverable: source code + dataset + manual so a reader can reproduce the work.

This file is the entry point for every coding agent. `CLAUDE.md` points here.

> Talk to the user in **Vietnamese**; write code, comments, commit messages and docs in **English**.
> The user learns while building: explain the *why* briefly, keep the layout obvious, record lessons in `docs/notes/`.

## Read before doing anything

| Doing what | Read |
|---|---|
| Anything | this file |
| The assignment text, grading, baseline facts | [`docs/reference/project-brief.md`](docs/reference/project-brief.md) (PDF next to it) |
| Naming anything | [`GLOSSARY.md`](GLOSSARY.md) |
| Adding or moving code | [`docs/reference/structure.md`](docs/reference/structure.md) |
| What's next | [`docs/plan/roadmap.md`](docs/plan/roadmap.md) |
| Why a choice was made | [`docs/ADR/`](docs/ADR) |
| Spending GPU credit (Modal) | [`docs/ops/compute-log.md`](docs/ops/compute-log.md) — log every run |

## Hard rules

1. **Fair comparison.** Every number is computed by `src/evaluate/metrics.py`, a line-for-line port of the
   baseline notebook metrics. Never change a metric to make a method look better.
2. **Fixed split.** `data/processed/split.json` (seeded) fixes `test` (≥ 100 questions, the number is reported),
   `dev`, `train`. Tune on `dev`, report on `test`, train/fine-tune only on `train`. Never read `test` answers
   while designing a method.
3. **LLM < 4B parameters.** Write the parameter count next to every model in `config/models.yaml`.
4. **Data stays out of git.** `data/raw/` and `data/processed/` are gitignored; `make data` rebuilds them.
5. **No GPU / Modal spend without the user's word.** Log every run in `docs/ops/compute-log.md`.
6. **Never print secrets** (`.env`: HF token, Modal profile).
7. Every merged change / result / decision adds a step to [`CHANGELOG.md`](CHANGELOG.md).

## Layout

```
src/            PYTHONPATH=src — map in docs/reference/structure.md
├─ cli/         one entrypoint per make command; the only place that wires parts
├─ corpus/      load the 16k law files, parse structure (Chương/Điều/Khoản), law ids, split QA
├─ index/       chunk → dense / BM25 / graph indexes
├─ retrieve/    query → ranked passages (dense, hybrid, rerank, graph expansion)
├─ generate/    prompt + LLM (HF transformers / vLLM), LoRA adapters
├─ evaluate/    baseline metrics (exact port) + retrieval metrics (law-id hit@k)
└─ shared/      config loading, paths, types
config/ models · retrieval · experiments/   notebooks/ (baseline copy, demo)   scripts/vendors/modal/
results/runs/<id>/ (predictions + metrics)   report/ (final write-up)   tests/unit/
```

## Commands

```bash
make            # list targets
make data       # unzip corpus, clean res.csv, write the fixed split
make test       # unit tests
make eval EXP=config/experiments/<x>.yaml   # run one experiment on dev or test, write results/runs/<id>/
```
