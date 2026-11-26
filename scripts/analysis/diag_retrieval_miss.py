"""Why does retrieval miss the gold article? Classifies every dev question by where the gold (law, Điều) pair is lost, from the cached retrieval of a config:
final top-k (dense -> rerank) and the dense candidate pool (rerank off, k = candidates; the cache the oracle mode builds).

  hit                          gold pair in the final top-k
  law ok, other article        gold law in the top-k, but not the gold Điều (sibling / amended article)
  dropped by rerank / k        gold pair in the dense pool (rank <=10 / <=20 / <=50) but not in the final top-k
  law in pool, article not     gold law in the pool, gold Điều never
  law not retrieved            gold law in the corpus but absent from the dense top-n  (embedding / query failure)
  law not in corpus            corpus gap
and for the last classes whether the question names a law number at all. Mean ROUGE-L of the run's answers per class when a run dir is given.

    PYTHONPATH=src python scripts/analysis/diag_retrieval_miss.py data/runs/A5_v3b__2e66c86a   (box)
"""
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from config import load
from corpus.keys import stem_key
from evaluate.metrics import rouge_l_score
from evaluate.retrieval_eval import RetrievalEval
from retrieve.retrieval import Retrieval

run = Path(sys.argv[1])
cfg = load("configs/exp/A5_v3b.yaml")   # retrieval settings of the A5_v3b runs (the retrieval cache key ignores the generator)
pool_cfg = cfg.model_copy(update={"rerank": cfg.rerank.model_copy(update={"model": None}), "retrieve": cfg.retrieve.model_copy(update={"k": cfg.retrieve.candidates, "oracle": False})})
final, pool = Retrieval(cfg)._read(), Retrieval(pool_cfg)._read()
ev = RetrievalEval()
qa = {r["qid"]: r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
split = json.load(open("data/processed/split.json"))
dev = set(split["dev"])
corpus = {stem_key(p.stem) for p in Path("data/processed/corpus").rglob("*.txt")}
answers = {}
if (run / "predictions.jsonl").exists():
    for line in open(run / "predictions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        answers[r["qid"]] = r["answer"]
LAWNO = re.compile(r"\d+\s*/\s*\d{4}\s*/\s*[A-ZĐ][\w\-]*|\d+\s*/\s*[A-ZĐ][\w\-]+")
cls, rl, namedlaw = {}, defaultdict(list), defaultdict(Counter)
for qid in sorted(dev & final.keys()):
    g = ev.gold.get(qid)
    if not (g and g.pairs):
        continue
    fk = [ev.passage_keys({"doc": p.doc, "text": p.text}) for p in final[qid]]
    pk = [ev.passage_keys({"doc": p.doc, "text": p.text}) for p in pool.get(qid, [])]
    pair = lambda k, a: a and f"{k}#{a}" in g.pairs
    if any(pair(k, a) for k, a in fk):
        c = "hit"
    elif any(k in g.docs for k, _ in fk):
        c = "law ok, other article"
    else:
        ranks = [i + 1 for i, (k, a) in enumerate(pk) if pair(k, a)]
        if ranks:
            r0 = ranks[0]
            c = "dropped by rerank/k (dense rank <=10)" if r0 <= 10 else "dropped by rerank/k (dense rank 11-20)" if r0 <= 20 else "dropped by rerank/k (dense rank 21-50)"
        elif any(k in g.docs for k, _ in pk):
            c = "law in pool, article not"
        elif any(d in corpus for d in g.docs):
            c = "law in corpus, not retrieved"
        else:
            c = "law not in corpus"
    cls[qid] = c
    if qid in answers:
        rl[c].append(rouge_l_score(answers[qid], qa[qid]["answer"]))
    named = bool(LAWNO.search(qa[qid]["question"]))
    namedlaw[c][named] += 1
n = len(cls)
print(f"dev questions with a gold (law, Điều): {n}")
print("| class | n | share | mean rougeL (run) | question names a law number |\n|---|---|---|---|---|")
for c, k in Counter(cls.values()).most_common():
    m = f"{st.mean(rl[c]):.3f}" if rl[c] else "-"
    print(f"| {c} | {k} | {k / n:.1%} | {m} | {namedlaw[c][True]}/{k} |")
