"""Why is ROUGE-L only ~.71 when the model cites the right Điều? Precision/recall of the LCS, length ratio and khoản agreement inside the 'retrieved and cited' group.

    PYTHONPATH=src python scripts/analysis/diag_cited_group.py data/runs/A5_v3b__127c207b [more runs]   (box)
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import polars as pl

from corpus.gold import GoldSet
from evaluate.faithfulness import dieu_numbers
from evaluate.metrics import normalize_text
from evaluate.retrieval_eval import RetrievalEval

KHOAN = re.compile(r"kho[aả]n\s+(\d+)", re.I)


def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def main():
    gold = GoldSet().load()
    qa = {str(r["qid"]): r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
    for run in map(Path, sys.argv[1:]):
        cache, _ = RetrievalEval.cache_of(run)
        passages = {json.loads(l)["qid"]: json.loads(l)["passages"] for l in cache.read_text(encoding="utf-8").splitlines()}
        rows = []
        for line in open(run / "predictions.jsonl", encoding="utf-8"):
            p = json.loads(line)
            g = gold.get(p["qid"])
            if not g or not g.pairs:
                continue
            pairs = {f"{k}#{a}" for k, a in (RetrievalEval.passage_keys(x) for x in passages.get(p["qid"], [])) if a}
            gn = {int(x.rsplit("#", 1)[1]) for x in g.pairs if x.rsplit("#", 1)[1].isdigit()}
            if not (set(g.pairs) & pairs and gn & dieu_numbers(p["answer"])):
                continue
            ref = qa[str(p["qid"])]["answer"] or ""
            a, r = normalize_text(p["answer"]).split(), normalize_text(ref).split()
            l = lcs(a, r)
            P, R = l / max(len(a), 1), l / max(len(r), 1)
            rk, ak = set(KHOAN.findall(ref)), set(KHOAN.findall(p["answer"]))
            rows.append(dict(P=P, R=R, F=2 * P * R / max(P + R, 1e-9), lr=len(a) / max(len(r), 1), ref_k=bool(rk), same_k=bool(rk) and rk <= ak, ngold=len(gn), ref_words=len(r)))
        f = lambda xs: round(st.mean(xs), 3)   # noqa: E731
        print(f"\n== {run.name}: cited-correct group n={len(rows)}  P {f([x['P'] for x in rows])}  R {f([x['R'] for x in rows])}  F {f([x['F'] for x in rows])}  median len ratio {round(st.median([x['lr'] for x in rows]), 2)}")
        for name, sel in [("answer >1.3x ref (over-quoting)", lambda x: x["lr"] > 1.3), ("answer <0.7x ref (under-quoting)", lambda x: x["lr"] < 0.7), ("0.7-1.3x", lambda x: 0.7 <= x["lr"] <= 1.3),
                          ("ref names a khoản", lambda x: x["ref_k"]), ("ref names no khoản", lambda x: not x["ref_k"]), ("ref khoản also in answer", lambda x: x["same_k"]),
                          ("ref khoản missing from answer", lambda x: x["ref_k"] and not x["same_k"]), ("1 gold article", lambda x: x["ngold"] == 1), (">=3 gold articles", lambda x: x["ngold"] >= 3),
                          ("ref < 80 words", lambda x: x["ref_words"] < 80), ("ref >= 250 words", lambda x: x["ref_words"] >= 250)]:
            s = [x for x in rows if sel(x)]
            if s and name.startswith(("answer >", "answer <")):
                print(f"      of which ref names a khoản: {sum(x['ref_k'] for x in s)}/{len(s)}")
            if s:
                print(f"  {name:36s} n={len(s):3d}  F {f([x['F'] for x in s])}  P {f([x['P'] for x in s])}  R {f([x['R'] for x in s])}")


if __name__ == "__main__":
    main()
