> **Superseded in part (2026-11-17):** the final layout keeps `src/` flat (no `lawrag` package), uses classes, and builds on LlamaIndex — see `docs/ADR/0002-llamaindex-and-layout.md` and `docs/reference/structure.md`.

# Clean codebase for law-rag — proposal (2026-11-17)

Inputs: findings_repo_layouts.md (FlashRAG, AutoRAG, Haystack, RAGLAB, cookiecutter-data-science),
findings_infra.md (config, tracking, eval harness, backends), findings_datamate_audit.md.

## Goal
A reader opens the repo and, in one screen, sees: the question flow, where each stage lives, how to run one experiment,
where its results land, and where the GPU runs. Size target: < 3k lines of src for the whole project.

## Layout

```
law-rag/
├─ AGENTS.md  README.md  GLOSSARY.md  CHANGELOG.md  Makefile  pyproject.toml
├─ configs/
│  ├─ base.yaml                 every default, commented — the one place to read all knobs
│  ├─ backends/                 WHERE the LLM / embedder runs (one file per box, not per model)
│  │  ├─ mac.yaml               mlx_lm / transformers, small smoke tests
│  │  ├─ l40s.yaml              vLLM via ssh tunnel, base_url http://localhost:8000/v1
│  │  └─ modal.yaml
│  └─ exp/                      WHAT is tested — diff vs base only, named <group><n>_<what>.yaml
│     ├─ B0_baseline.yaml       course notebook as-is
│     ├─ C3_dieu_chunks.yaml
│     └─ R3_hybrid_rrf.yaml …
├─ src/lawrag/                  installable package (uv), no PYTHONPATH tricks
│  ├─ types.py                  Question, Passage, Answer, RunMeta — the only shared records
│  ├─ config.py                 yaml → pydantic (extra="forbid"), base + exp + --set a.b=c
│  ├─ registry.py               dict name → class, one per stage (≈30 lines)
│  ├─ data/        load.py  parse.py (Chương/Điều/Khoản)  split.py  decontam.py
│  ├─ index/       chunk.py  dense.py  bm25.py           build once, saved under data/index/<hash>/
│  ├─ retrieve/    dense.py  bm25.py  hybrid.py  rerank.py   Retriever.retrieve(q, k) -> list[Passage]
│  ├─ generate/    llm.py (one OpenAI-compatible client)  prompts/*.md  postprocess.py
│  ├─ evaluate/    metrics.py (exact baseline port)  retrieval.py (hit@k)  bootstrap.py
│  ├─ pipeline.py  question → retrieve → rerank → generate, each stage cached on disk
│  ├─ runs.py      run dir + results.tsv ledger
│  └─ cli.py       lawrag data | index | run | score | compare | demo
├─ train/          later: sft.py, raft.py, configs/ — kept out of src until needed
├─ ops/            how to drive GPUs (not science code)
│  ├─ l40s/  sync.sh  serve.sh  run.sh  status.sh  README.md
│  └─ modal/ app.py
├─ notebooks/      explore only (EDA, error analysis); nothing imported from here
├─ tests/          unit (metrics, chunking, config, registry) + smoke (5 questions, fake LLM)
├─ docs/           ADR/  notes/  reference/  plan/  ops/compute-log.md
├─ report/         final write-up
├─ data/   (gitignored) raw/ processed/ index/
└─ runs/   (gitignored) <date>_<exp>_<hash>/ + results.tsv
```

## Conventions (each from a source or a datamate lesson)

| # | Rule | Why |
|---|---|---|
| 1 | Stages = folders under `src/lawrag/`, implementations = **files**, not sub-folders | datamate: 106 `__init__.py`, 10 adapter packages for 4.5k lines |
| 2 | One small interface per stage + dict registry; config picks by name (`retriever: hybrid`) | FlashRAG / Haystack; datamate registry (48 lines) |
| 3 | Typed records only in `types.py`; no `extra` grab-bag dict | datamate `Context.extra` was undocumented data flow |
| 4 | Config = YAML + pydantic, not Hydra; `base.yaml` holds all defaults, exp files only the diff | Hydra chdir / override grammar pain; datamate diff-only experiments were the clean part |
| 5 | Backend = one OpenAI-compatible client, `base_url` from `configs/backends/*.yaml` | same code on Mac / L40S / Modal; datamate had 21 near-duplicate runtime yamls |
| 6 | Run dir `runs/<date>_<exp>_<hash>/`: config.yaml, meta.json (git sha, model revisions, backend), retrieved.jsonl, predictions.jsonl, metrics.json; resumable | datamate runs.py; autoresearch |
| 7 | `runs/results.tsv` one row per run (exp, hash, n, 5 metrics, hit@5, status, note) | Karpathy autoresearch; datamate strict LedgerRow |
| 8 | Stage cache: retrieval cached separately from generation → swap LLM without re-retrieving | eval-harness guides |
| 9 | Scoring separate: `lawrag score <run>` re-reads predictions; paired bootstrap in `compare` | evaluator separation; significance guides |
| 10 | Files ≤ ~300 lines; no `utils.py`; heavy imports inside functions | datamate 500–1,045-line files |
| 11 | `ops/` scripts drive boxes; science code never knows which box it is on | datamate scripts/ overlapped adapter/launch |
| 12 | One-off analysis → notebook or `docs/notes/NN-*.md`, never a new script in src | datamate 62 one-off scripts, 145 research docs |

## Migration from today's repo (~550 lines)
- `src/{corpus,index,retrieve,generate,evaluate,shared,cli}` → `src/lawrag/{data,index,retrieve,generate,evaluate}` + `types.py`/`config.py`/`cli.py`.
- `config/experiments/baseline.yaml` → `configs/exp/B0_baseline.yaml` + `configs/base.yaml`.
- `scripts/vendors/modal/app.py` → `ops/modal/app.py`; new `ops/l40s/`.
- `results/runs/` → `runs/`; `make` targets call `lawrag <cmd>`.
- Update AGENTS.md, docs/reference/structure.md, GLOSSARY.md.
