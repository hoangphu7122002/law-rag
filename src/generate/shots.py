"""Few-shot examples: the nearest decontaminated train questions (token Jaccard) with their reference answers.

    ShotStore().pick(question, n)  ->  [(question, answer), ...]   never the question itself
The pool is the decontaminated train split (AGENTS.md rule 3), so a shot can never be a test pair; the same store is used
for inference and for building SFT rows, so the two formats cannot drift.
"""
import re

from corpus.decontam import Decontam
from corpus.qa import QASet
from evaluate.retrieval_eval import RetrievalEval

WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text)}


class ShotStore:
    def __init__(self, pool=None):
        pool = pool if pool is not None else Decontam().filter(QASet().load("train"))[0]
        gold = RetrievalEval().gold
        self.pool = [(q.text, q.answer.strip(), words(q.text), frozenset(g.pairs) if (g := gold.get(q.qid)) else frozenset())
                     for q in pool]

    def pick(self, question: str, n: int, avoid: frozenset = frozenset()) -> list[tuple[str, str]]:
        """avoid = gold (law#Điều) pairs of the row being built: a train row must not see a shot citing its own Điều
        (a test question never can, the pool is decontaminated), else the model learns to copy the shot."""
        if n <= 0:
            return []
        w = words(question)
        scored = sorted(((len(w & x) / max(1, len(w | x)), t, a) for t, a, x, pr in self.pool
                         if t != question and not (avoid and pr & avoid)), reverse=True)
        return [(t, a) for _, t, a in scored[:n]]
