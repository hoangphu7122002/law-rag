# Findings: RAG component class design (2026-11-17)

Source: research subagent (read FlashRAG, LlamaIndex BaseRetriever, Haystack component, DSPy Module sources).

| Library | Base | Selection | Composition | Loading | Weak points |
|---|---|---|---|---|---|
| FlashRAG | plain `BaseRetriever._search/_batch_search`; BaseRetriever→BaseTextRetriever→Dense/BM25 | if/elif + importlib in `get_retriever(config)` | `SequentialPipeline(config, retriever=None, generator=None)` inject-or-build; `run(dataset)` | in `__init__` | one global config dict; twin search paths; auto-detect magic |
| LlamaIndex | ABC `BaseRetriever._retrieve(QueryBundle)`; public `retrieve` = template method | built in code | `RetrieverQueryEngine(retriever, synthesizer, postprocessors)` | `__init__` | deep hierarchy, wrapper types, callbacks |
| Haystack 2 | no base, `@component` + `run()`; `warm_up()` for heavy load | YAML type path registry | graph `add_component` + `connect` | `warm_up()` | sockets/metaclass machinery |
| DSPy | `Module.forward`, `__call__` wraps it | in code; global `dspy.configure` | attributes are sub-modules | — | global settings |

## Recommendations
1. One verb per noun: Retriever.search, Reranker.rerank, Generator.generate, Embedder.embed, Pipeline.run, Evaluator.score.
2. Batch-first signatures, no twin single/batch methods.
3. Protocol or ABC for the interfaces; keep hierarchy flat (no middle layers).
4. Cheap `__init__`, lazy `_load()` on first use.
5. Config → class via small explicit dict per stage, typed per-component config.
6. Pipeline = plain class with injected components; returns one result per question.
7. Avoid: shared global config dict, loading in __init__, decorators/metaclasses/graph wiring, wrapper types, global settings, async twins, registry decorators.

Sources: raw.githubusercontent.com/RUC-NLPIR/FlashRAG/main/flashrag/{retriever/retriever.py,pipeline/pipeline.py,utils/utils.py} ·
run-llama/llama_index .../base_retriever.py · deepset-ai/haystack .../component.py · stanfordnlp/dspy .../module.py
