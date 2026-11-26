# src/train/retrieval/

retrieval-side fine-tuning: embedder and reranker data and trainers.

Import as `train.retrieval.<module>`; run as `python -m train.retrieval.<module>` or `python src/train/retrieval/<module>.py` with `PYTHONPATH=src`.

| module | what |
|---|---|
| `embed_ft.py` | Fine-tune the embedder (AITeamVN/Vietnamese_Embedding) on the rerank-data groups, then score it with EmbedBench. |
| `rerank_data.py` | Reranker fine-tune data: (query, positive, hard negatives) groups, decontaminated against test. |
| `rerank_ft.py` | Fine-tune the cross-encoder reranker on src/train/retrieval/rerank_data.py groups (sentence-transformers CrossEncoderTrainer). |
