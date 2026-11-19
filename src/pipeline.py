"""The RAG pipeline and one experiment run.

RAGPipeline   question + passages -> Prediction (one question; failures become an empty answer + error, never a crash)
Experiment    config -> questions -> Retrieval (cached) -> RAGPipeline over all questions -> Run dir -> Evaluator
"""
import asyncio
import time

from config import Config
from corpus import QASet
from evaluate import Evaluator
from generate import FakeGenerator, Generator
from records import Passage, Prediction, Question
from retrieve import Retrieval
from runs import Run


class RAGPipeline:
    def __init__(self, generator: Generator):
        self.generator = generator

    async def answer(self, question: Question, passages: list[Passage]) -> Prediction:
        t = time.time()
        docs = [p.doc for p in passages]
        try:
            text = await self.generator.answer(question.text, passages)
            return Prediction(question.qid, text, docs, time.time() - t)
        except Exception as e:  # noqa: BLE001 — one bad question must not stop a 300-question run
            return Prediction(question.qid, "", docs, time.time() - t, error=repr(e)[:300])


class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run = Run.create(cfg)

    def execute(self, score: bool = True) -> dict | None:
        questions = QASet().load(self.cfg.split, self.cfg.limit)
        passages = Retrieval(self.cfg).run(questions)
        todo = [q for q in questions if q.qid not in self.run.done_qids()]
        print(f"{self.run.path}: {len(questions)} questions, {len(todo)} to answer")
        asyncio.run(self._answer_all(todo, passages))
        fake = self.cfg.backend.name == "fake"  # no sentence-transformers needed for smoke runs
        return Evaluator(cosine=not fake, device=self.cfg.backend.device).score(self.run) if score else None

    async def _answer_all(self, questions: list[Question], passages: dict[int, list[Passage]]) -> None:
        g = self.cfg.generate
        if g.agent and self.cfg.backend.name != "fake":
            from generate.react import LawTools, ReactGenerator

            generator = ReactGenerator(g, self.cfg.backend, LawTools(self.cfg))
        else:
            generator = (FakeGenerator if self.cfg.backend.name == "fake" else Generator)(g, self.cfg.backend)
        pipeline = RAGPipeline(generator)
        limit = asyncio.Semaphore(g.concurrency)

        async def one(q: Question) -> None:
            async with limit:
                pred = await pipeline.answer(q, passages[q.qid])
            self.run.append_prediction(pred.to_dict())

        await asyncio.gather(*(one(q) for q in questions))
