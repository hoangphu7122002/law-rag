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


class LowScoreReranker:   # nothing in the corpus relates to the question: every cross-encoder score is ~0
    def postprocess_nodes(self, nodes, query_bundle=None):
        return [NodeWithScore(node=n.node, score=0.0002) for n in nodes]


def test_off_topic_question_is_refused_without_calling_the_llm():
    class NoLLM:
        async def answer(self, question, passages):
            raise AssertionError("the LLM must not be called")

    svc = QAService(FakeRetriever(), LowScoreReranker(), NoLLM(), k=2, min_score=0.001)
    out = asyncio.run(svc.ask("Hôm nay thời tiết ở Hà Nội thế nào?"))
    assert out["refused"] is True and "chưa tìm thấy" in out["answer"].lower()
    assert out["timings"]["generate_s"] == 0.0 and len(out["sources"]) == 2


def test_relevant_question_is_not_refused_and_default_guard_is_off():
    out = asyncio.run(QAService(FakeRetriever(), ReverseReranker(), FakeGenerator(), k=2, min_score=0.001).ask("q"))
    assert out["refused"] is False
    assert asyncio.run(QAService(FakeRetriever(), LowScoreReranker(), FakeGenerator(), k=2).ask("q"))["refused"] is False


class StreamingGenerator(FakeGenerator):
    async def stream(self, question, passages):
        for piece in ["Xin ", "chào ", "bạn"]:
            yield piece


async def collect(agen):
    return [e async for e in agen]


def test_stream_emits_sources_then_tokens_then_done_with_ttft():
    svc = QAService(FakeRetriever(), ReverseReranker(), StreamingGenerator(), k=2)
    events = asyncio.run(collect(svc.stream("câu hỏi?")))
    assert [e["event"] for e in events] == ["sources", "token", "token", "token", "done"]
    assert [s["doc"] for s in events[0]["sources"]] == ["C", "B"]
    assert "".join(e["text"] for e in events if e["event"] == "token") == "Xin chào bạn"
    done = events[-1]
    assert done["answer"] == "Xin chào bạn" and done["refused"] is False
    assert 0 <= done["timings"]["ttft_s"] <= done["timings"]["total_s"]


def test_stream_falls_back_to_one_token_when_generator_cannot_stream():
    events = asyncio.run(collect(service(k=1).stream("q")))
    assert [e["event"] for e in events] == ["sources", "token", "done"]
    assert events[1]["text"] == events[-1]["answer"]


def test_stream_refuses_below_threshold_without_calling_the_generator():
    class Boom(FakeGenerator):
        async def answer(self, *a):
            raise AssertionError("must not be called")

    svc = QAService(FakeRetriever(), ReverseReranker(), Boom(), k=2, min_score=5.0)
    events = asyncio.run(collect(svc.stream("off topic")))
    assert events[-1]["refused"] is True and events[-1]["answer"].startswith("Dựa trên")
