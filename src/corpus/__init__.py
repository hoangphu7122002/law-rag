"""Raw data in, clean objects out.

DatasetBuilder   `make data`: unzip the corpus, clean the RES sheet, write the fixed split
QASet            the questions of one split
sources/         CorpusReader, VbplImporter, ExtraData — where documents come from
ingest/          LawCleaner, DieuNodeParser — LlamaIndex transforms, Document -> nodes
"""
from corpus.builder import DatasetBuilder
from corpus.ingest import DieuNodeParser, LawCleaner
from corpus.qa import QASet
from corpus.sources import CorpusReader

__all__ = ["CorpusReader", "DatasetBuilder", "DieuNodeParser", "LawCleaner", "QASet"]
