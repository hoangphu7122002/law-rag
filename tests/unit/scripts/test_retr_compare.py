import json
import sys
from pathlib import Path

from retr_compare import per_question_hits


class Gold:
    def __init__(self, pairs):
        self.docs, self.pairs = {p.split("#")[0] for p in pairs}, set(pairs)


def line(qid, docs_arts):
    return json.dumps({"qid": qid, "passages": [{"doc": d, "text": f"Điều {a}. nội dung"} for d, a in docs_arts]})


def test_per_question_art_hit_at_k(tmp_path):
    cache = tmp_path / "c.jsonl"
    cache.write_text("\n".join([line(1, [("A", 1), ("A", 2)]), line(2, [("B", 9)]), line(3, [("C", 1)])]) + "\n")
    gold = {1: Gold({"A#2"}), 2: Gold({"B#1"}), 3: Gold(set())}      # q3 has no gold article: skipped
    hits = per_question_hits(cache, gold, k=5, level="art")
    assert hits == {1: 1.0, 2: 0.0}
    assert per_question_hits(cache, gold, k=1, level="art") == {1: 0.0, 2: 0.0}
    assert per_question_hits(cache, gold, k=1, level="doc") == {1: 1.0, 2: 1.0}
