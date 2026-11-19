"""Test-like dev subsets (2026-11-19). Plain dev shares a Điều with the (test-decontaminated) train for 45 % of its
questions and a non-boilerplate 13-gram for 72 %, while test shares 0 Điều and 32 % → dev over-states template/shot gains.

  dev_clean   dev questions whose gold (law key, Điều) pairs are not in the kept train   (n = 166)
  dev_strict  dev_clean minus questions sharing a non-boilerplate answer 13-gram with kept train   (n = 74)

    DevSets().build()  -> data/processed/dev_sets.json   (QASet.load("dev_clean" | "dev_strict") reads it)
"""
import json
from collections import Counter
from pathlib import Path

from corpus.decontam import BOILERPLATE, Decontam, grams
from corpus.gold import GoldSet
from corpus.qa import QASet


class DevSets:
    def __init__(self, processed: Path = Path("data/processed")):
        self.processed = Path(processed)

    def build(self) -> dict:
        qa, gold = QASet(self.processed), GoldSet().load()
        train, _ = Decontam(self.processed).filter(qa.load("train"))
        dev, test = qa.load("dev"), qa.load("test")
        pairs = set().union(*(gold[q.qid].pairs for q in train if q.qid in gold))
        seen = Counter(g for q in train + dev + test for g in grams(q.answer))
        tg = {g for q in train for g in grams(q.answer) if seen[g] <= BOILERPLATE}
        clean = [q for q in dev if not (q.qid in gold and gold[q.qid].pairs & pairs)]
        strict = [q for q in clean if not grams(q.answer) & tg]
        out = {"dev_clean": [q.qid for q in clean], "dev_strict": [q.qid for q in strict]}
        (self.processed / "dev_sets.json").write_text(json.dumps(out))
        print({k: len(v) for k, v in out.items()})
        return out
