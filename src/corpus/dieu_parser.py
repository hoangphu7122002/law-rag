"""One node per Điều (article). Long articles are split again by size; each piece keeps the article heading.

The corpus is a Word export: paragraphs are separated by blank lines and lines inside a paragraph are hard-wrapped.
About 75 % of files have line-start headings "Điều N." (500 random files checked); the rest become one
preamble section that is split by size.
"""
import re
from collections.abc import Sequence

from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode
from pydantic import Field

HEADING = re.compile(r"(?m)^Điều\s+\d+[a-zđ]?\s*[.:]")


class DieuNodeParser(NodeParser):
    max_chars: int = Field(default=2000, description="articles longer than this are split")
    overlap: int = Field(default=100)

    @classmethod
    def class_name(cls) -> str:
        return "DieuNodeParser"

    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs) -> list[BaseNode]:
        out: list[BaseNode] = []
        for doc in nodes:
            pieces = [p for heading, body in self.split_articles(doc.get_content()) for p in self._fit(heading, body)]
            out.extend(build_nodes_from_splits(pieces, doc, id_func=self.id_func))
        return out

    @staticmethod
    def unwrap(text: str) -> str:
        """Join hard-wrapped lines inside a paragraph; one newline between paragraphs."""
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


