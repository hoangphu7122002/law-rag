"""Raw data in, clean objects out.

DatasetBuilder   `make data`: unzip the corpus, clean the RES sheet, write the fixed split
QASet            the questions of one split
CorpusReader     law files -> LlamaIndex Documents
DieuNodeParser   LlamaIndex node parser: one node per Điều (article)
"""
from corpus.builder import DatasetBuilder
from corpus.dieu_parser import DieuNodeParser
from corpus.qa import QASet
from corpus.reader import CorpusReader

__all__ = ["CorpusReader", "DatasetBuilder", "DieuNodeParser", "QASet"]
