"""Question -> ranked passages.

RetrieverFactory   config -> a LlamaIndex retriever (dense | bm25 | hybrid RRF) and an optional reranker
Retrieval          runs them over a question set, cached as data/cache/retrieval/<key>.jsonl
"""
from retrieve.factory import RetrieverFactory
from retrieve.retrieval import Retrieval

__all__ = ["Retrieval", "RetrieverFactory"]
