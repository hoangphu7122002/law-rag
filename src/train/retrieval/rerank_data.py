"""Reranker fine-tune data: (query, positive, hard negatives) groups, decontaminated against test.

Sources
  course   train questions (Decontam) x their top-`candidates` DENSE hits (no rerank, cached by Retrieval):
           positive = a hit whose (law key, Điều) is a gold pair; negatives = hits of other Điều (same document first,
           the hardest), never a gold pair, never a same-document chunk without a heading (may be the gold Điều),
           never one scoring > 95 % of the positive (NV-Retriever TopK-PercPos false-negative filter).
           Questions with no positive among the candidates are skipped (reported).
  zalo     Zalo AI legal retrieval 2021 (data/raw/extra/zalo): question -> article "<law>#<Điều>"; negatives = other
           articles of the same law, then random articles. Pairs whose article is cited by test are dropped, and
           questions sharing a non-boilerplate 13-gram with a test answer.
  val      decontaminated dev, course-style, for the in-training reranking evaluator (MRR@10 / NDCG@10).

    RerankData(cfg, "rr_v0").build()  -> data/train/rr_v0/{train.jsonl, val.jsonl, report.json}
    rows: {"query", "pos": [text], "neg": [text, ...]}   (val: "docs" = all candidates, "pos" = the gold ones)
"""
import json
import random
from pathlib import Path

from config import Config
from corpus.decontam import Decontam, grams
from corpus.keys import law_key
from corpus.qa import QASet
from evaluate.retrieval_eval import RetrievalEval
from records import Passage
from retrieve import Retrieval

OUT, ZALO = Path("data/train"), Path("data/raw/extra/zalo")
PERC_POS = 0.95


class RerankData:
    def __init__(self, cfg: Config, name: str, negatives: int = 7, zalo: bool = True, seed: int = 20261031):
        # candidates = the dense top-N the reranker sees at inference (rerank off, k = candidates)
        self.cfg = cfg.model_copy(update={"rerank": cfg.rerank.model_copy(update={"model": None}),
                                          "retrieve": cfg.retrieve.model_copy(update={"k": cfg.retrieve.candidates})})
        self.name, self.negatives, self.zalo, self.rng = name, negatives, zalo, random.Random(seed)
        self.out = OUT / name

    def build(self) -> Path:
        dc, ev = Decontam(), RetrievalEval()
        train, report = dc.filter(QASet().load("train"))
        dev, vrep = dc.filter(QASet().load("dev"))
        rows, skipped = self._course(train, ev)
        report |= {"course_rows": len(rows), "course_no_positive": skipped}
        if self.zalo:
            z, zrep = self._zalo(dc)
            rows += z
            report["zalo"] = zrep
        val, vskip = self._course(dev, ev, val=True)
        report["val"] = {"in": vrep["in"], "kept": vrep["kept"], "rows": len(val), "no_positive": vskip}
        self.rng.shuffle(rows)
        self.out.mkdir(parents=True, exist_ok=True)
        for name, data in (("train", rows), ("val", val)):
            with open(self.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        report |= {"retrieval_key": self.cfg.retrieval_key(), "candidates": self.cfg.retrieve.k, "negatives": self.negatives}
        Decontam.save(report, self.out / "report.json")
        print(f"rerank data: {len(rows)} train groups, {len(val)} val queries -> {self.out}")
        return self.out

    def _course(self, questions, ev: RetrievalEval, val: bool = False) -> tuple[list[dict], int]:
        hits = Retrieval(self.cfg).run(questions)   # dense top-N, cached
        rows, skipped = [], 0
        for q in questions:
            g = ev.gold.get(q.qid)
            if not g or not g.pairs:
                skipped += 1
                continue
            pos, same_doc, other = [], [], []
            for p in hits[q.qid]:
                key, art = ev.passage_keys({"doc": p.doc, "text": p.text})
                if art and f"{key}#{art}" in g.pairs:
                    pos.append(p)
                elif key in g.docs:
                    (same_doc if art else []).append(p)   # same document, other Điều; headless chunks dropped
                else:
                    other.append(p)
            if not pos:
                skipped += 1
                continue
            # NV-Retriever TopK-PercPos: a negative scoring > 95 % of the positive is likely a false negative
            cap = PERC_POS * max(p.score for p in pos)
            same_doc, other = ([p for p in ps if p.score <= cap] for ps in (same_doc, other))
            pos, same_doc, other = ([p.text for p in ps] for ps in (pos, same_doc, other))
            if val:
                rows.append({"query": q.text, "pos": pos, "docs": [p.text for p in hits[q.qid]]})
            else:
                rows.append({"query": q.text, "pos": pos[:1], "neg": (same_doc + other)[: self.negatives]})
        return rows, skipped

    def _zalo(self, dc: Decontam) -> tuple[list[dict], dict]:
        import polars as pl

        queries = dict(pl.read_parquet(ZALO / "queries/train-00000-of-00001.parquet").iter_rows())
        corpus = pl.read_parquet(ZALO / "corpus/train-00000-of-00001.parquet")
        text = {i: f"{t}\n{x}" for i, t, x in corpus.iter_rows()}
        by_law: dict[str, list[str]] = {}
        for i in text:
            by_law.setdefault(i.split("#")[0], []).append(i)
        ids = list(text)
        qrels = pl.concat([pl.read_parquet(ZALO / f"qrels/{s}-00000-of-00001.parquet") for s in ("train", "test")])
        rows, drop = [], {"test_article": 0, "ngram13": 0}
        for cid, qid, _ in qrels.iter_rows():
            law, art = cid.split("#")
            if f"{law_key(law)}#{art}" in dc.test_pairs:
                drop["test_article"] += 1
                continue
            if grams(queries[qid]) & dc.test_grams or grams(text[cid]) & dc.test_grams:
                drop["ngram13"] += 1
                continue
            same = [i for i in by_law[law] if i != cid]
            neg = self.rng.sample(same, min(len(same), self.negatives - 2))
            while len(neg) < self.negatives:
                r = self.rng.choice(ids)
                if r != cid and r not in neg:
                    neg.append(r)
            rows.append({"query": queries[qid], "pos": [text[cid]], "neg": [text[i] for i in neg]})
        rep = {"in": qrels.height, "kept": len(rows), "dropped": drop}
        print(f"zalo: {rep}")
        return rows, rep
