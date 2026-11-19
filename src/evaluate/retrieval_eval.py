"""Retrieval quality without the LLM: did the passages contain the cited document / Điều?

    RetrievalEval().score(cache, qids) -> {n, doc@1, doc@3, …, art@k, doc_mrr}
    python -m cli retrieval data/runs/<run>   (or a data/cache/retrieval/<key>.jsonl)

doc@k  = share of questions whose top-k passages include any gold document (corpus.gold, law keys)
art@k  = … include a gold (document, Điều) pair; the Điều of a passage is the first "Điều N." line of its text
Only questions with a gold document (resp. Điều) count, so numbers are comparable across corpora.
Scores use LlamaIndex's HitRate / MRR retrieval metrics with law keys (doc) and "<key>#<Điều>" (art) as ids.
"""
import json
import re
from pathlib import Path

from llama_index.core.evaluation.retrieval.metrics import MRR, HitRate

from corpus.gold import GoldSet
from corpus.keys import stem_key

HEADING = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zđ]?)\s*[.:]")
KS = (1, 3, 5, 10, 20)


class RetrievalEval:
    def __init__(self, gold: dict | None = None):
        self.gold = gold if gold is not None else GoldSet().load()

    @staticmethod
    def passage_keys(p: dict) -> tuple[str, str | None]:
        """(law key, Điều or None) of one cached passage."""
        m = HEADING.search(p.get("text", "")[:400])
        return stem_key(p["doc"]), (m.group(1) if m else None)

    def score_one(self, qid: int, passages: list[dict]) -> dict | None:
        g = self.gold.get(qid)
        if not g or not g.docs:
            return None
        keys = [self.passage_keys(p) for p in passages]
        docs = [k for k, _ in keys]
        arts = [f"{k}#{a}" if a else f"{k}#?" for k, a in keys]
        row = {"qid": qid, "has_art": bool(g.pairs)}
        for name, ids, gold in (("doc", docs, g.docs), ("art", arts, g.pairs)):
            for k in KS:
                row[f"{name}@{k}"] = HitRate().compute(expected_ids=list(gold), retrieved_ids=ids[:k]).score if gold else 0.0
            row[f"{name}_mrr"] = MRR().compute(expected_ids=list(gold), retrieved_ids=ids).score if gold else 0.0
        return row

    def score(self, cache: Path, qids: set[int] | None = None) -> dict:
        rows = []
        for line in Path(cache).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if qids is None or r["qid"] in qids:
                row = self.score_one(r["qid"], r["passages"])
                if row:
                    rows.append(row)
        return self.summary(rows)

    @staticmethod
    def summary(rows: list[dict]) -> dict:
        arts = [r for r in rows if r["has_art"]]
        out = {"n": len(rows), "n_art": len(arts)}
        for name, rs in (("doc", rows), ("art", arts)):
            for m in [f"{name}@{k}" for k in KS] + [f"{name}_mrr"]:
                out[m] = round(sum(r[m] for r in rs) / max(len(rs), 1), 4)
        return out

    @staticmethod
    def cache_of(run: Path) -> tuple[Path, set[int]]:
        """Run dir -> (its retrieval cache, its qids)."""
        from config import Config
        from retrieve.retrieval import CACHE
        from runs import Run

        r = Run(Path(run))
        cfg = Config(**r.config())
        return CACHE / f"{cfg.retrieval_key()}.jsonl", r.done_qids()
