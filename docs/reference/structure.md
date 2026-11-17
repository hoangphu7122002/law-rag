# Structure

`src/` is on `PYTHONPATH` (pytest and the Makefile set it); nothing is installed. Flow of one experiment:

```
configs/exp/X.yaml ─► config.load() ─► Config
                                         │
Experiment.execute()                     ▼
  QASet.load(split) ─► questions
  Retrieval.run(questions) ──── cache: data/cache/retrieval/<retrieval_key>.jsonl
      └─ RetrieverFactory ─► LlamaIndex retriever (dense | bm25 | hybrid) [+ reranker]
            └─ IndexStore ─── cache: data/index/<chunk>__<index_key>/ (docstore.json, faiss/)
                  └─ CorpusReader ─► Chunker (LangChain recursive | DieuNodeParser)
  RAGPipeline.answer(question, passages) ─► Generator ─► Prompt + OpenAILike (vLLM / mlx / Modal)
      └─ Run.append_prediction ─► data/runs/<exp>__<run_key>/predictions.jsonl  (rerun = resume)
  Evaluator.score(run) ─► metrics.json + Ledger row in data/runs/results.tsv
```

## Classes

| Class | File | Job | Uses |
|---|---|---|---|
| `Config` (+ `Chunk`, `Embed`, `Retrieve`, `Rerank`, `Generate`, `Backend`) | `config.py` | validated settings; `index_key` / `retrieval_key` / `run_key` hashes | pydantic |
| `Question`, `Passage`, `Prediction` | `records.py` | the only records passed between stages | — |
| `DatasetBuilder` | `corpus/builder.py` | `make data` | polars |
| `QASet` | `corpus/qa.py` | questions of a split | — |
| `CorpusReader` | `corpus/reader.py` | law files → `Document` | LlamaIndex |
| `DieuNodeParser(NodeParser)` | `corpus/dieu_parser.py` | one node per Điều | LlamaIndex |
| `Chunker` | `index/chunker.py` | picks the node parser | LlamaIndex, LangChain splitter |
| `IndexStore` | `index/store.py` | nodes + FAISS index cache | LlamaIndex, FAISS, HF embeddings |
| `RetrieverFactory` | `retrieve/factory.py` | `retrieve.kind` → retriever; reranker | LlamaIndex BM25 / QueryFusion / SBERT rerank |
| `Retrieval` | `retrieve/retrieval.py` | run + cache retrieval | — |
| `Prompt` | `generate/prompt.py` | prompt file → messages / completion text (shared with SFT data) | — |
| `Generator`, `FakeGenerator` | `generate/generator.py` | LLM call + cleanup | LlamaIndex OpenAILike |
| `RAGPipeline`, `Experiment` | `pipeline.py` | one question; one whole run | everything above |
| `Run`, `Ledger` | `runs.py` | run folder; `data/runs/results.tsv` | — |
| `RemoteBox` | `ops/box.py` | push code/data, setup, serve vLLM, run, status, stop, pull (ssh + rsync + tmux) | `make box` |
| `Evaluator`, `PairedBootstrap` | `evaluate/` | scores, law hit@k; significance | metrics (exact port) |

## Rules

| Folder | May import |
|---|---|
| `records.py`, `config.py`, `runs.py` | nothing in `src/` except each other |
| `corpus/` | records |
| `index/` | config, corpus |
| `retrieve/` | config, index, records, `evaluate.metrics.normalize_text` |
| `generate/` | config, records |
| `evaluate/` | corpus, runs |
| `ops/` | `config`, `runs` only |
| `pipeline.py`, `cli.py` | everything (only `cli.py` imports `ops`) |

Adding a method = a new class or a new `kind` in the factory + a key in `configs/base.yaml` + an experiment yaml.
Training (`train/`, TRL + PEFT) comes later and reuses `Prompt` for its data.
