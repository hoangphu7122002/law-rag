"""Where documents come from.

CorpusReader   law files (course corpus, vbpl) -> LlamaIndex Documents
VbplImporter   `make vbpl`: vbpl.vn central documents -> data/processed/vbpl
ExtraData      `make extra`: public legal datasets for training -> data/raw/extra
"""
from corpus.sources.extra import ExtraData
from corpus.sources.reader import CorpusReader
from corpus.sources.vbpl import VbplImporter

__all__ = ["CorpusReader", "ExtraData", "VbplImporter"]
