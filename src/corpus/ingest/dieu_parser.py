"""One node per Điều (article). Long articles are split again by size; each piece keeps the article heading.

The corpus is a Word export: paragraphs are separated by blank lines and lines inside a paragraph are hard-wrapped.
About 75 % of files have line-start headings "Điều N." (500 random files checked); the rest become one
preamble section that is split by size.

Node metadata (hidden from embedding and LLM text, like the document's): article ("5", "" for the preamble),
heading ("Điều 5. Phân loại …"), chapter ("Chương II. …", the last chapter heading before the article).
"""
import re
from collections.abc import Sequence

from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode
from pydantic import Field

HEADING = re.compile(r"(?m)^Điều\s+(\d+[a-zđ]?)\s*[.:]")
CHAPTER = re.compile(r"(?m)^Chương\s+[IVXLC\d]+\b.*$")
NODE_KEYS = ["article", "heading", "chapter"]


class DieuNodeParser(NodeParser):
    max_chars: int = Field(default=2000, description="articles longer than this are split")
    overlap: int = Field(default=100)

    @classmethod
    def class_name(cls) -> str:
        return "DieuNodeParser"

    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs) -> list[BaseNode]:
        out: list[BaseNode] = []
        for doc in nodes:
            pieces, meta, chapter = [], [], ""
            for heading, body in self.split_articles(doc.get_content()):
                m = HEADING.match(heading)
                info = {"article": m.group(1) if m else "", "heading": heading, "chapter": chapter}
                for piece in self._fit(heading, body):
                    pieces.append(piece)
                    meta.append(info)
                chapters = CHAPTER.findall(body)  # a chapter heading ends the previous article's body
                chapter = chapters[-1].strip()[:200] if chapters else chapter
            for node, info in zip(build_nodes_from_splits(pieces, doc, id_func=self.id_func), meta, strict=True):
                node.metadata.update(info)
                node.excluded_embed_metadata_keys = [*node.excluded_embed_metadata_keys, *NODE_KEYS]
                node.excluded_llm_metadata_keys = [*node.excluded_llm_metadata_keys, *NODE_KEYS]
                out.append(node)
        return out

    @staticmethod
    def unwrap(text: str) -> str:
        """Join hard-wrapped lines inside a paragraph; one newline between paragraphs.
        Text without blank lines (vbpl, LawCleaner output) is already one paragraph per line: kept as is."""
        if not re.search(r"\n\s*\n", text):
            return text.strip()
        joined = (re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text))
        return "\n".join(p for p in joined if p)

    @classmethod
    def split_articles(cls, text: str) -> list[tuple[str, str]]:
        """[(heading, body)]; heading "" for the preamble. The body starts with its heading line."""
        text = cls.unwrap(text)
        starts = [m.start() for m in HEADING.finditer(text)]
        if not starts:
            return [("", text)] if text else []
        parts = [("", text[: starts[0]].strip())] if text[: starts[0]].strip() else []
        for a, b in zip(starts, starts[1:] + [len(text)]):
            body = text[a:b].strip()
            parts.append((body.split("\n", 1)[0][:200], body))
        return parts

    def _fit(self, heading: str, body: str) -> list[str]:
        if len(body) <= self.max_chars:
            return [body]
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_chars, chunk_overlap=self.overlap)
        pieces = splitter.split_text(body)
        return [pieces[0]] + [f"{heading}\n{p}" if heading else p for p in pieces[1:]]


