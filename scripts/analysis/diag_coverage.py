"""Does the context (retrieved chunks) contain what the reference quotes? Reference 4-gram coverage by (a) the context, (b) the full gold articles (upper bound),
split by the length of the longest gold article. Also: how much of the gold articles' text is in the context (chunk-level, not article-level).

    PYTHONPATH=src python scripts/analysis/diag_coverage.py data/runs/A5_v3b__127c207b   (box)
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

from corpus.gold import GoldSet
from evaluate.metrics import normalize_text
from evaluate.retrieval_eval import RetrievalEval


def grams(t, n=4):
    w = normalize_text(t).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    run = Path(sys.argv[1])
    A = {}
    for l in open("data/cpt/articles.jsonl", encoding="utf-8"):
        r = json.loads(l)
        k, a = r["key"].split("#", 1)
        A[f"{k.upper()}#{a.lower()}"] = r["text"]
    gold = GoldSet().load()
    qa = {str(r["qid"]): r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
    cache, _ = RetrievalEval.cache_of(run)
    passages = {json.loads(l)["qid"]: json.loads(l)["passages"] for l in cache.read_text(encoding="utf-8").splitlines()}
    scores = {r["qid"]: r["rouge_l"] for r in json.load(open(run / "metrics.json"))["rows"]}
    B = defaultdict(list)
    for q, s in scores.items():
        g = gold.get(q)
        if not g or not g.pairs:
            continue
        ref = grams(qa[str(q)]["answer"] or "")
        full = [A.get(f"{p.rsplit('#', 1)[0].upper()}#{p.rsplit('#', 1)[1].lower()}") for p in g.pairs]
        full = [x for x in full if x]
        if not ref or not full:
            continue
        ctx = set().union(*[grams(p["text"]) for p in passages.get(q, [])]) if passages.get(q) else set()
        fullg = set().union(*[grams(x) for x in full])
        cov_ctx = len(ref & ctx) / len(ref)
        cov_full = len(ref & fullg) / len(ref)
        art_in_ctx = len(fullg & ctx) / max(len(fullg), 1)   # share of the gold articles' text present in the context
        m = max(len(x) for x in full)
        b = "<1000" if m < 1000 else ("1000-2000" if m <= 2000 else (">2000-4000" if m <= 4000 else ">4000"))
        B[b].append((cov_ctx, cov_full, art_in_ctx, s))
        B["all"].append((cov_ctx, cov_full, art_in_ctx, s))
    print(f"{run.name}: reference 4-gram coverage by the context vs by the full gold articles; share of gold-article text in the context")
    for k in ["<1000", "1000-2000", ">2000-4000", ">4000", "all"]:
        v = B[k]
        print(f"  longest gold article {k:11s} n={len(v):3d}  ref covered by context {st.mean(x[0] for x in v):.3f}  by full gold {st.mean(x[1] for x in v):.3f}  gold text in context {st.mean(x[2] for x in v):.3f}  rouge_l {st.mean(x[3] for x in v):.3f}")


if __name__ == "__main__":
    main()
