"""Test decontamination of every training set (AGENTS.md rule 3). One module, one report per training set.

A train / dev pair is dropped when it
  - is a test pair (never in the training split anyway, checked),
  - cites a (law key, Điều) that a test answer cites,
  - shares a non-boilerplate word 13-gram with a test answer (boilerplate = 13-gram in > 3 answers overall).

    kept, report = Decontam().filter(train_questions)
"""
import json
import re
from collections import Counter
from pathlib import Path

from records import Question

N, BOILERPLATE = 13, 3


def grams(text: str, n: int = N) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", (text or "").lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


class Decontam:
    def __init__(self, processed: Path = Path("data/processed")):
        from corpus.gold import GoldSet
        from corpus.qa import QASet

        qa = QASet(processed)
        self.test = qa.load("test")
        self.test_ids = {q.qid for q in self.test}
        gold = GoldSet().load()
        self.test_pairs = set().union(*(gold[q.qid].pairs for q in self.test if q.qid in gold))
        self.gold = gold
        every = [q for s in ("train", "dev", "test") for q in qa.load(s)]
        seen = Counter(g for q in every for g in grams(q.answer))
        self.test_grams = {g for q in self.test for g in grams(q.answer) if seen[g] <= BOILERPLATE}

    def reason(self, q: Question) -> str | None:
        if q.qid in self.test_ids:
            return "test_pair"
        g = self.gold.get(q.qid)
        if g and g.pairs & self.test_pairs:
            return "same_law_article"
        if grams(q.answer) & self.test_grams or grams(q.text) & self.test_grams:
            return "ngram13"
        return None

    def filter(self, questions: list[Question]) -> tuple[list[Question], dict]:
        why = {q.qid: self.reason(q) for q in questions}
        kept = [q for q in questions if why[q.qid] is None]
        report = {"in": len(questions), "kept": len(kept), "dropped": dict(Counter(r for r in why.values() if r)),
                  "dropped_qids": sorted(k for k, r in why.items() if r), "test_pairs": len(self.test_pairs),
                  "test_grams": len(self.test_grams)}
        print(f"decontam: {report['in']} -> {report['kept']} ({report['dropped']})")
        return kept, report

    @staticmethod
    def save(report: dict, path: Path) -> None:
        Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=1))
