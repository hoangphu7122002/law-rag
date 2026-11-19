# Structure

`src/` is on `PYTHONPATH` (pytest and the Makefile set it); nothing is installed. Flow of one experiment:

```
configs/exp/X.yaml ─► config.load() ─► Config
                                         │
Experiment.execute()                     ▼
  QASet.load(split) ─► questions         (split: train | dev | dev_clean | dev_strict | test)
  Retrieval.run(questions) ──── cache: data/cache/retrieval/<retrieval_key>.jsonl
      └─ RetrieverFactory ─► LlamaIndex retriever (dense | bm25 | hybrid) [+ reranker]
            └─ IndexStore ─── cache: data/index/<chunk>__<index_key>/ (docstore.json, faiss/)
                  └─ CorpusReader ─► Chunker (LangChain recursive | DieuNodeParser)
  RAGPipeline.answer(question, passages) ─► Generator ─► Prompt + OpenAILike (vLLM / mlx / Modal)
      │                     └─ optional: shots · MBR (+ gate) · ReactGenerator (tools: search_law, get_article, kg_related, web_search)
      └─ Run.append_prediction ─► data/runs/<exp>__<run_key>/predictions.jsonl  (rerun = resume)
  Evaluator.score(run) ─► metrics.json + Ledger row in data/runs/results.tsv
```

## Classes

| Class | File | Job | Uses |
|---|---|---|---|
| `Config` (+ `Chunk`, `Embed`, `Retrieve`, `Rerank`, `Generate`, `WebCfg`, `Backend`, `Box`) | `config.py` | validated settings; `index_key` / `retrieval_key` / `run_key` hashes (a new knob at its default leaves old keys unchanged) | pydantic |
| `Question`, `Passage`, `Prediction` | `records.py` | the only records passed between stages | — |
| `DatasetBuilder` | `corpus/builder.py` | `make data` | polars |
| `QASet` | `corpus/qa.py` | questions of a split (`dev_clean` / `dev_strict` read `dev_sets.json`) | — |
| `DevSets` | `corpus/devsets.py` | test-like dev subsets (no Điều / 13-gram shared with train) | decontam |
| `Decontam`, `GoldSet`, `law_key` | `corpus/decontam.py`, `gold.py`, `keys.py` | test decontamination of every training set; gold (law, Điều) per question; comparable law keys | — |
| `CorpusReader`, vbpl, `LuatVietnam`, `Citations` | `corpus/sources/` | law files, vbpl.vn dataset, luatvietnam.vn crawl, cited-document table | LlamaIndex, polars |
| `DieuNodeParser(NodeParser)`, cleaner, dedup | `corpus/ingest/` | one node per Điều; boilerplate removal; document dedup | LlamaIndex |
| `Chunker` | `index/chunker.py` | picks the node parser | LlamaIndex, LangChain splitter |
| `IndexStore` | `index/store.py` | nodes + FAISS index cache | LlamaIndex, FAISS, HF / vLLM embeddings |
| `RetrieverFactory` | `retrieve/factory.py` | `retrieve.kind` → retriever; reranker (local or vLLM) | LlamaIndex BM25 / QueryFusion / SBERT rerank |
| `Retrieval` | `retrieve/retrieval.py` | run + cache retrieval | — |
| `Prompt`, `ShotStore` | `generate/prompt.py`, `shots.py` | prompt file → messages / completion text (shared with SFT data); nearest train Q/A shots | — |
| `Generator`, `FakeGenerator` | `generate/generator.py` | LLM call, cleanup, MBR pick (+ optional gate) | LlamaIndex OpenAILike |
| `Gate` | `generate/check.py` | optional answer gate (`generate.check`), e.g. grounded citation | — |
| `ReactGenerator`, `LawTools` | `generate/react.py` | ReAct loop on native tool calls with loop guards; live search / article / KG / web tools | openai client, retrieve, kg |
| `WebSearch`, `TinyFish` | `generate/web.py` | web search tool: cache, domain allow-list, leak filter, rate and budget limits | TinyFish Search API |
| `KG` | `kg/store.py` | rule-built legal KG in SQLite (vbpl relationships + validity); hop-limited `related`; `render_related` | polars (build only) |
| `SFTData`, `RerankData` | `train/sft_data.py`, `rerank_data.py` | SFT rows (k-range, oracle, shots, refusal rows), reranker groups | corpus, retrieve, generate |
| `sft_lora`, `rerank_ft`, `grpo` | `train/*.py` | scripts run in the box's training envs (TRL + PEFT; vLLM colocate for GRPO) | torch, TRL, PEFT |
| `RAGPipeline`, `Experiment` | `pipeline.py` | one question; one whole run | everything above |
| `Run`, `Ledger` | `runs.py` | run folder; `data/runs/results.tsv` | — |
| `RemoteBox` | `ops/box.py` | push code/data, setup, serve vLLM (`--tools`), queue DSL (`sh` / `wait` / `do` / `serve`), run, status, stop, pull (ssh + rsync + tmux; `.rl_env` and `data/` are never synced) | `make box` |
| `Evaluator`, `PairedBootstrap` | `evaluate/` | scores, law / Điều hit@k; significance | metrics (exact port) |

## Rules (imports as they are today, checked with grep 2026-11-19)

| Folder | May import |
|---|---|
| `records.py`, `config.py`, `runs.py` | nothing in `src/` except each other |
| `corpus/` | records (and itself) |
| `index/` | config, corpus |
| `retrieve/` | config, index, records, `evaluate.metrics.normalize_text` |
| `generate/` | config, records, corpus (keys, decontam, qa), evaluate (metrics), kg, retrieve / index (live tools only, inside methods) |
| `kg/` | corpus.keys (build only) |
| `evaluate/` | corpus, runs |
| `train/` | config, corpus, evaluate, generate, records, retrieve |
| `ops/` | `config`, `runs` only |
| `pipeline.py`, `cli.py` | everything (only `cli.py` imports `ops`) |

Science code never imports `ops`. Heavy imports (torch, vLLM clients, LlamaIndex retrievers, polars) stay inside methods.

## Outside `src/`

| Path | What |
|---|---|
| `configs/base.yaml`, `configs/exp/*.yaml`, `configs/backends/*.yaml` | every knob; one diff-only yaml per experiment; machine settings |
| `configs/queues/*.txt` | box queue files, one per queue (`make box ARGS="queue configs/queues/<name>.txt --gpu N --parallel"`): the exact jobs queued on the L40S |
| `docs/notes/` | numbered notes (what was learned) |
| `docs/research/<topic>/` | web-research output: `research_plan.md`, `findings_*.md` |
| `docs/ops/compute-log.md`, `docs/CHANGELOG.md` | every GPU run, every step |

Adding a method = a new class or a new `kind` in the factory + a key in `configs/base.yaml` + an experiment yaml, with the test written first (`tdd`).
