"""Documents -> nodes, chosen by config.chunk.kind."""
from llama_index.core.node_parser import LangchainNodeParser, NodeParser
from llama_index.core.schema import BaseNode, Document

from config import Chunk
from corpus import DieuNodeParser


class Chunker:
    def __init__(self, cfg: Chunk):
        self.cfg = cfg

    def parser(self) -> NodeParser:
        if self.cfg.kind == "dieu":
            return DieuNodeParser(max_chars=self.cfg.size, overlap=self.cfg.overlap)
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # same splitter as the course baseline

        lc = RecursiveCharacterTextSplitter(chunk_size=self.cfg.size, chunk_overlap=self.cfg.overlap)
        return LangchainNodeParser(lc, include_prev_next_rel=False)

    def split(self, documents: list[Document]) -> list[BaseNode]:
        return self.parser().get_nodes_from_documents(documents, show_progress=True)
