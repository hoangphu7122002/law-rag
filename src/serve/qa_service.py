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

    async def _prepare(self, question: str):
        """Retrieve + rerank + refusal decision; returns (passages, refused, t0, t1, t2)."""
        t0 = time.perf_counter()
        hits = await asyncio.to_thread(self.retriever.retrieve, question)      # blocking HTTP + FAISS: keep the event loop free
        t1 = time.perf_counter()
        if self.reranker is not None:
            hits = await asyncio.to_thread(self.reranker.postprocess_nodes, hits, QueryBundle(question))
        t2 = time.perf_counter()
        passages: list[Passage] = to_passages(hits[: self.k])
        refused = bool(passages) and self.reranker is not None and passages[0].score < self.min_score
        return passages, refused, t0, t1, t2

    def _sources(self, passages) -> list[dict]:
        return [{"doc": p.doc, "text": p.text, "score": p.score} for p in passages]

    async def stream(self, question: str):
        """Events for a streaming UI: sources (once retrieval is done), token (text pieces), done (full answer + timings incl. ttft_s)."""
        passages, refused, t0, t1, t2 = await self._prepare(question)
        yield {"event": "sources", "sources": self._sources(passages)}
        parts: list[str] = []
        ttft = None
        if refused:
            parts.append(REFUSAL)
            yield {"event": "token", "text": REFUSAL}
            ttft = time.perf_counter() - t0
        elif hasattr(self.generator, "stream"):
            async for piece in self.generator.stream(question, passages):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(piece)
                yield {"event": "token", "text": piece}
        else:
            text = await self.generator.answer(question, passages)
            ttft = time.perf_counter() - t0
            parts.append(text)
            yield {"event": "token", "text": text}
        t3 = time.perf_counter()
        total = round(t3 - t0, 3)
        yield {"event": "done", "answer": "".join(parts), "refused": refused,
               "timings": {"retrieve_s": round(t1 - t0, 3), "rerank_s": round(t2 - t1, 3) if self.reranker is not None else 0.0,
                           "generate_s": round(t3 - t2, 3), "ttft_s": round(min(ttft if ttft is not None else total, total), 3), "total_s": total}}

    async def ask(self, question: str) -> dict:
        passages, refused, t0, t1, t2 = await self._prepare(question)
        answer = REFUSAL if refused else await self.generator.answer(question, passages)
        t3 = time.perf_counter()
        return {"answer": answer, "refused": refused,
                "sources": self._sources(passages),
                "timings": {"retrieve_s": round(t1 - t0, 3), "rerank_s": round(t2 - t1, 3) if self.reranker is not None else 0.0,
                            "generate_s": round(t3 - t2, 3), "total_s": round(t3 - t0, 3)}}
