"""Documents -> nodes with a LlamaIndex IngestionPipeline: [LawCleaner] -> [DocDedup] -> node parser (by config.chunk.kind)."""
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import LangchainNodeParser, NodeParser
from llama_index.core.schema import BaseNode, Document, TransformComponent

from config import Chunk
from corpus import DieuNodeParser, LawCleaner
from corpus.ingest import DocDedup


class Chunker:
    def __init__(self, cfg: Chunk):
        self.cfg = cfg

    def parser(self) -> NodeParser:
        if self.cfg.kind == "dieu":
            return DieuNodeParser(max_chars=self.cfg.size, overlap=self.cfg.overlap)
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # same splitter as the course baseline

        lc = RecursiveCharacterTextSplitter(chunk_size=self.cfg.size, chunk_overlap=self.cfg.overlap)
        return LangchainNodeParser(lc, include_prev_next_rel=False)

    def transformations(self) -> list[TransformComponent]:
        return ([LawCleaner()] if self.cfg.clean else []) + ([DocDedup()] if self.cfg.dedup else []) + [self.parser()]

    def split(self, documents: list[Document]) -> list[BaseNode]:
        return IngestionPipeline(transformations=self.transformations()).run(documents=documents, show_progress=True)
