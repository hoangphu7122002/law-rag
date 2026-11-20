from types import SimpleNamespace

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from retrieve.factory import RetrieverFactory


class _Fixed(BaseRetriever):
    def __init__(self, ids):
        super().__init__()
        self.ids = ids

    def _retrieve(self, query_bundle):
        return [NodeWithScore(node=TextNode(id_=i, text=i), score=1.0 / (n + 1)) for n, i in enumerate(self.ids)]


def test_hybrid_fuses_without_any_llm_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = SimpleNamespace(retrieve=SimpleNamespace(kind="hybrid", candidates=3, k=2), rerank=SimpleNamespace(model=None))
    f = RetrieverFactory(cfg, store=None)
    f.dense = lambda n: _Fixed(["a", "b", "c"])
    f.bm25 = lambda n: _Fixed(["b", "a", "d"])
    got = [n.node.node_id for n in f.retriever().retrieve("q")]
    assert set(got) >= {"a", "b"} and len(got) <= 3
