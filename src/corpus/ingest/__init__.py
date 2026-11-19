"""LlamaIndex ingestion transforms (used in an IngestionPipeline by index.Chunker).

LawCleaner       TransformComponent: Document -> cleaned Document (+ title, type, status … metadata)
DocDedup         TransformComponent: drop exact / same-number near-duplicate Documents
DieuNodeParser   NodeParser: one node per Điều (article)
"""
from corpus.ingest.cleaner import LawCleaner
from corpus.ingest.dedup import DocDedup
from corpus.ingest.dieu_parser import DieuNodeParser

__all__ = ["DieuNodeParser", "DocDedup", "LawCleaner"]
