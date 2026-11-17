# ADR 0002 — LlamaIndex as the RAG framework; class-based flat `src/` layout

Date: 2026-11-17. Status: accepted.

## Context
- The roadmap needs hybrid retrieval, rerankers, Qdrant, Neo4j graphs, query rewriting (HyDE, multi-query),
  CRAG, agents / workflows, Mem0 memory, RAFT data, vLLM serving. Writing each by hand costs weeks.
- The datamate codebase (25k lines) showed what hurts readability: 106 `__init__.py`, 21 near-duplicate runtime yamls,
  62 one-off scripts, files over 1,000 lines, a free-form `Context.extra` dict. What worked: diff-only experiment
  configs, a resumable run folder, a strict tsv ledger, per-question failure isolation.
- Research notes: `docs/notes/06-codebase-design/` (repo layouts, infra, datamate audit, class patterns).

## Decision
1. **LlamaIndex** for RAG parts. Checked on PyPI 2026-11-17: `retrievers-bm25`, `vector-stores-qdrant`,
   `graph-stores-neo4j`, `postprocessor-sbert-rerank`, `llms-openai-like`, `llms-vllm`, `packs-raft-dataset`,
   `packs-corrective-rag`, `memory-mem0`, Workflows — all present. Haystack lacks several (Neo4j community package
   stale, no RAFT/CRAG packs); FlashRAG is method-rich but hard to extend; LangChain APIs churn.
2. **TRL + PEFT** for training, **vLLM** for serving (OpenAI-compatible endpoint, same client everywhere).
3. **Classes, flat `src/`**: one folder per stage, one class per job, at most one inheritance level, lazy model loading.
4. **YAML + pydantic** config (`configs/base.yaml` + diff-only `configs/exp/*.yaml`), not Hydra.
5. **Own thin code** only where no framework fits or fairness requires it: config, run folder + ledger,
   the exact baseline metrics, the Điều parser (as a LlamaIndex `NodeParser`), decontamination.

## Consequences
- New methods = a new `kind` in `RetrieverFactory` (or a LlamaIndex class) + a config key + an experiment yaml.
- LightRAG / MS GraphRAG / Graphiti stay separate libraries behind a thin adapter, whatever the framework.
- LlamaIndex docstore JSON for ~190k nodes is large (hundreds of MB) and slow to load; move to Qdrant (K1) if it hurts.
