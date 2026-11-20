"""Few-shot examples: the nearest decontaminated train questions (token Jaccard) with their reference answers.

    ShotStore().pick(question, n)  ->  [(question, answer), ...]   never the question itself
The pool is the decontaminated train split (AGENTS.md rule 3), so a shot can never be a test pair; the same store is used
for inference and for building SFT rows, so the two formats cannot drift.
"""
import json
import re
from pathlib import Path

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

    def save(self, path) -> None:
        """Write the pool to JSON so a service can ship it without the dataset (gold pairs are kept: `avoid` still works)."""
        rows = [{"q": t, "a": a, "pairs": sorted(map(list, pr))} for t, a, _, pr in self.pool]
        Path(path).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ShotStore":
        store = cls.__new__(cls)   # skip the dataset / decontamination build
        store.pool = [(r["q"], r["a"], words(r["q"]), frozenset(map(tuple, r["pairs"])))
                      for r in json.loads(Path(path).read_text(encoding="utf-8"))]
        return store
