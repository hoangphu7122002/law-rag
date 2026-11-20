"""QAService: question -> dense retrieval -> cross-encoder rerank -> greedy generation, with per-stage timings.

The three heavy parts are injected (a LlamaIndex retriever + reranker that call the embedder / reranker endpoints, and the
Generator that calls the LLM endpoint), so the service itself holds no model and is tested with fakes.
"""
import asyncio
import time

from llama_index.core import QueryBundle

from records import Passage
from retrieve.retrieval import to_passages

REFUSAL = "Dựa trên các ngữ cảnh được cung cấp, chưa tìm thấy quy định liên quan để trả lời câu hỏi này."   # same wording as the SFT refusal rows


class QAService:
    def __init__(self, retriever, reranker, generator, k: int = 5, min_score: float = 0.0):
        """min_score: if the best reranker score is below it the question has no support in the corpus: refuse instead of letting the LLM improvise (0 = off)."""
        self.retriever, self.reranker, self.generator, self.k, self.min_score = retriever, reranker, generator, k, min_score

    async def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        hits = await asyncio.to_thread(self.retriever.retrieve, question)      # blocking HTTP + FAISS: keep the event loop free
        t1 = time.perf_counter()
        if self.reranker is not None:
            hits = await asyncio.to_thread(self.reranker.postprocess_nodes, hits, QueryBundle(question))
        t2 = time.perf_counter()
        passages: list[Passage] = to_passages(hits[: self.k])
        refused = bool(passages) and self.reranker is not None and passages[0].score < self.min_score
        answer = REFUSAL if refused else await self.generator.answer(question, passages)
        t3 = time.perf_counter()
        return {"answer": answer, "refused": refused,
                "sources": [{"doc": p.doc, "text": p.text, "score": p.score} for p in passages],
                "timings": {"retrieve_s": round(t1 - t0, 3), "rerank_s": round(t2 - t1, 3) if self.reranker is not None else 0.0,
                            "generate_s": round(t3 - t2, 3), "total_s": round(t3 - t0, 3)}}
