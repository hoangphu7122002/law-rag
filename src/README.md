# src/

| package | purpose |
|---|---|
| `corpus/` | sources, ingestion, article (Điều) parsing, gold sets, decontamination |
| `index/` | chunking and the vector store |
| `retrieve/` | dense retrieval, reranking, gate / widen |
| `generate/` | prompts, generators, MBR, ReAct and tools |
| `train/` | SFT, preference optimisation, CPT, ReAct, retrieval fine-tuning (see [train/README.md](train/README.md)) |
| `evaluate/` | metrics, retrieval eval, faithfulness, bootstrap |
| `kg/` | law-reference knowledge graph |
| `serve/` | QA API |
| `ops/` | GPU box, Modal apps |
| `cli.py config.py pipeline.py records.py runs.py` | entry point, config loading, the RAG pipeline, run records |
