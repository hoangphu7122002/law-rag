"""Run retrieval (+ rerank) for every question once; later runs with the same key read the cache.

Cache: data/cache/retrieval/<retrieval_key>.jsonl, one line {qid, passages: [{pid, doc, text, score}]}.
Swapping the LLM or the prompt does not change the key, so retrieval is not repeated.
"""
import json
from dataclasses import asdict
from pathlib import Path

from llama_index.core import QueryBundle
from tqdm import tqdm

from config import Config
from index import IndexStore
from records import Passage, Question
from retrieve.factory import RetrieverFactory

CACHE = Path("data/cache/retrieval")


class Retrieval:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = CACHE / f"{cfg.retrieval_key()}.jsonl"

    def run(self, questions: list[Question]) -> dict[int, list[Passage]]:
        cached = self._read()
        todo = [q for q in questions if q.qid not in cached]
        if todo:
            factory = RetrieverFactory(self.cfg, IndexStore(self.cfg))
            retriever, reranker = factory.retriever(), factory.reranker()
            for q in tqdm(todo, desc="retrieve"):
                cached[q.qid] = self._one(q, retriever, reranker)
                self._append(q.qid, cached[q.qid])
        return {q.qid: cached[q.qid] for q in questions}

    def _one(self, q: Question, retriever, reranker) -> list[Passage]:
        hits = retriever.retrieve(q.text)
        if reranker is not None:
            hits = reranker.postprocess_nodes(hits, query_bundle=QueryBundle(q.text))
        hits = hits[: self.cfg.retrieve.k]
        return [Passage(pid=h.node.node_id, doc=h.node.metadata.get("doc", ""), text=h.node.get_content(),
                        score=float(h.score or 0.0)) for h in hits]

    def _read(self) -> dict[int, list[Passage]]:
        if not self.path.exists():
            return {}
        out = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[row["qid"]] = [Passage(**p) for p in row["passages"]]
        return out

    def _append(self, qid: int, passages: list[Passage]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"qid": qid, "passages": [asdict(p) for p in passages]}, ensure_ascii=False) + "\n")
