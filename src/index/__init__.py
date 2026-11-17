"""Corpus -> nodes (chunks) -> indexes, built once and cached under data/index/<key>/.

Chunker      Documents -> nodes, by config (recursive characters | one per Điều)
IndexStore   owns the cache folder: nodes (docstore) and the FAISS vector index
"""
from index.chunker import Chunker
from index.store import IndexStore

__all__ = ["Chunker", "IndexStore"]
