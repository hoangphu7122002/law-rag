import asyncio
from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

from serve.qa_service import QAService


def node(doc: str, text: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, metadata={"doc": doc}), score=score)


class FakeRetriever:
    def __init__(self):
        self.asked = []

    def retrieve(self, question):
        self.asked.append(question)
        return [node("A", "Điều 1. nội dung A", 0.9), node("B", "Điều 2. nội dung B", 0.8), node("C", "Điều 3. nội dung C", 0.7)]


class ReverseReranker:   # the cross-encoder disagrees with the dense order
    def postprocess_nodes(self, nodes, query_bundle=None):
        return [NodeWithScore(node=n.node, score=1.0 - i / 10) for i, n in enumerate(reversed(nodes))]


class FakeGenerator:
    async def answer(self, question, passages):
        return f"{question} -> " + ",".join(p.doc for p in passages)


def service(k=2, reranker=ReverseReranker()):
    return QAService(FakeRetriever(), reranker, FakeGenerator(), k=k)


def test_ask_retrieves_reranks_keeps_k_and_answers():
    out = asyncio.run(service(k=2).ask("câu hỏi?"))
    assert out["answer"] == "câu hỏi? -> C,B"                                 # reranked order, top-2 only
    assert [s["doc"] for s in out["sources"]] == ["C", "B"]
    assert "Điều 3. nội dung C" in out["sources"][0]["text"] and out["sources"][0]["score"] == 1.0


def test_ask_reports_stage_timings():
    t = asyncio.run(service().ask("q"))["timings"]
    assert set(t) == {"retrieve_s", "rerank_s", "generate_s", "total_s"}
    assert t["total_s"] >= t["generate_s"] >= 0


def test_ask_without_reranker_keeps_dense_order():
    out = asyncio.run(service(k=2, reranker=None).ask("q"))
    assert [s["doc"] for s in out["sources"]] == ["A", "B"] and out["timings"]["rerank_s"] == 0.0
