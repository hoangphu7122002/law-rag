"""DocDedup — LlamaIndex TransformComponent: drop duplicate Documents before parsing (IngestionPipeline step).

1. exact: same text after NFC, lower-case and dropping everything but letters / digits
2. near:  same law key (corpus.keys.stem_key of metadata["doc"]) and word 5-gram Jaccard >= threshold.
   Same number + different text (other year, other issuer: "02/NQ-CP" of 2019 vs 2024) is kept.
The first document wins, so list sources by trust in the config: corpus, vbpl_cited, web.
"""
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from hashlib import blake2b

from llama_index.core.schema import BaseNode, TransformComponent
from pydantic import Field

from corpus.keys import stem_key


class DocDedup(TransformComponent):
    threshold: float = Field(default=0.6, description="5-gram Jaccard at or above which same-key documents are copies")
    shingle: int = Field(default=5)

    @classmethod
    def class_name(cls) -> str:
        return "DocDedup"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> list[BaseNode]:
        keys = [stem_key(n.metadata.get("doc", "")) for n in nodes]
        shared = {k for k, c in Counter(keys).items() if k and c > 1}  # shingles only for numbers seen twice (RAM)
        seen, by_key, out, dropped = set(), {}, [], {"exact": 0, "near": 0}
        for node, key in zip(nodes, keys, strict=True):
            words = self.words(node.get_content())
            h = blake2b(" ".join(words).encode(), digest_size=16).digest()
            if h in seen:
                dropped["exact"] += 1
                continue
            seen.add(h)
            if key in shared:
                grams = self.shingles(words)
                if any(self.jaccard(grams, g) >= self.threshold for g in by_key.get(key, [])):
                    dropped["near"] += 1
                    continue
                by_key.setdefault(key, []).append(grams)
            out.append(node)
        print(f"dedup: {len(nodes)} -> {len(out)} documents (exact {dropped['exact']}, near {dropped['near']})")
        return out

    @staticmethod
    def words(text: str) -> list[str]:
        return re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())

    def shingles(self, words: list[str]) -> set[int]:
        n = self.shingle
        return {hash(tuple(words[i:i + n])) for i in range(max(len(words) - n + 1, 1))}

    @staticmethod
    def jaccard(a: set, b: set) -> float:
        return len(a & b) / max(len(a | b), 1)
