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
| Naming anything | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) |
| Adding or moving code | [`docs/reference/structure.md`](docs/reference/structure.md) |
| What's next | [`docs/plan/roadmap.md`](docs/plan/roadmap.md) |
| Why a choice was made | [`docs/ADR/`](docs/ADR) |
| Using a GPU (L40S box or Modal) | [`docs/ops/l40s.md`](docs/ops/l40s.md), [`docs/ops/compute-log.md`](docs/ops/compute-log.md) — log every run |

## Hard rules

1. **Fair comparison.** Every number is computed by `src/evaluate/metrics.py`, a line-for-line port of the
   baseline notebook metrics. Never change a metric to make a method look better.
2. **Fixed split.** `data/processed/split.json` (seeded) fixes `test` (200, reported), `dev` (300), `train` (2,245).
   Tune on `dev`, report on `test`. Never read `test` answers while designing a method.
3. **No test knowledge in any weights.** Nothing the model is *trained* on (pretraining corpus, SFT, RAFT, DPO, GRPO,
   embedding / reranker fine-tune) may contain knowledge tied to `test` questions:
   - drop the test QA pairs themselves;
   - drop every train/dev pair citing the same (law id, Điều) as a test answer;
   - drop every train/dev pair or crawled passage sharing a non-boilerplate 13-gram with a test answer
     (boilerplate = 13-gram found in > 3 answers);
   - drop the crawled text of every (law id, Điều) cited by test.
   `dev` pairs stay out of training **while tuning** (they choose hyper-parameters and checkpoints); the final model is
   retrained on train + dev with the chosen settings. The **RAG store may contain everything** (full laws, crawled
   data, train/dev QA) except the test QA pairs themselves.
   The filter lives in one module (`src/corpus/decontam.py`) and every training set records the filter report.
4. **LLM < 4B parameters.** Write the parameter count next to every model in `config/models.yaml`.
5. **Data stays out of git.** `data/raw/` and `data/processed/` are gitignored; `make data` rebuilds them.
6. **No GPU run (L40S or Modal) without the user's word.** The L40S box is shared: touch only `/home/app/phunguyen/law_rag`
   and tmux sessions `lawrag-*`. Log every run in `docs/ops/compute-log.md`.
7. **Never print secrets** (`.env`: HF token, Modal profile).
8. Every merged change / result / decision adds a step to [`docs/CHANGELOG.md`](docs/CHANGELOG.md).
9. **Test first.** New code is built with the `tdd` skill: one failing test through the public interface, then the minimal code, repeat; fake only external boundaries (LLM client, HTTP, GPU).
   A new tool / module without its tests does not count as done.

## Layout

```
src/        CODE — on PYTHONPATH; class map in docs/reference/structure.md
├─ corpus/    DatasetBuilder · QASet · CorpusReader · DieuNodeParser
├─ index/     Chunker · IndexStore (nodes + FAISS, cached in data/index/<key>/)
├─ retrieve/  RetrieverFactory (dense | bm25 | hybrid + reranker) · Retrieval (cached in data/cache/retrieval/)
├─ generate/  Prompt (prompts/*.md) · Generator (OpenAI-compatible LLM, MBR) · ReactGenerator (tools) · WebSearch · Gate · FakeGenerator
├─ evaluate/  metrics (exact port) · Evaluator · PairedBootstrap
├─ kg/        KG (rule-built legal graph, SQLite) · render_related
├─ train/     SFT / reranker data + LoRA scripts · grpo.py (run in the box's training envs)
├─ pipeline.py RAGPipeline · Experiment        config.py  Config + load()
├─ records.py  Question · Passage · Prediction  runs.py    Run · Ledger
├─ serve/      QAService · REST api (POST /ask, GET /health) · wiring — the QA container (docker/, docs/ops/docker.md)
├─ ops/        RemoteBox (ssh + rsync + tmux: push, setup, serve vLLM, run, status, stop, pull) · modal_app.py
└─ cli.py      one subcommand per make target
tests/      CHECKS — unit/ (no data, no models) · smoke/ (whole pipeline, BM25 + fake LLM)
configs/    WHAT TO RUN — base.yaml (every knob) · exp/<group><n>_<what>.yaml (diff only) · backends/{l40s,mac,fake}.yaml · queues/*.txt (box job queues)
data/       INPUTS + OUTPUTS (gitignored except runs/results.tsv) — raw/ processed/ index/ cache/
            runs/<exp>__<hash>/ {config.yaml, meta.json, predictions.jsonl, metrics.json} · runs/results.tsv (ledger)
docs/       TO READ — GLOSSARY · CHANGELOG · ADR/ · plan/ · reference/ · notes/ · research/ (web-research output) · ops/ · notebooks/ (explore only) · report/
```

Framework: **LlamaIndex** for RAG parts (node parsers, FAISS index, BM25, query fusion, rerank, OpenAI-like LLM),
**vLLM** to serve, **TRL + PEFT** to train. Our own code stays thin: config, runs, metrics, Điều parser, decontamination.

## Code style

- Classes with one job each; one level of inheritance at most (e.g. `DieuNodeParser(NodeParser)`).
- Heavy models load on first use, never at import or in a cheap constructor; heavy imports inside methods.
- Files ≤ ~300 lines; no `utils.py`; new knob → `configs/base.yaml` with a comment, never a hidden default.
- Science code never knows which machine it runs on: only `configs/backends/*.yaml` and `src/ops/` do; science modules never import `ops`.

## Commands

```bash
make              # list targets
make data         # unzip corpus, clean res.csv, write the fixed split
make test         # unit + smoke tests (smoke needs `make data`)
make smoke        # whole pipeline on 5 questions: BM25 + fake LLM, no GPU
make run EXP=configs/exp/B0_baseline.yaml [ARGS="--set retrieve.k=10 --backend mac"]
make score RUN=data/runs/<dir>          make compare A=data/runs/<a> B=data/runs/<b> METRIC=rouge_l
```
