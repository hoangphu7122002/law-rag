"""How far can an EXTRACTIVE answer (template + chosen clause of the gold article) get, versus the model? Dev questions whose reference names a khoản of a multi-clause gold article.
Variants: oracle clause (the one the reference names), lexical clause selector top-1 / top-1+2 / top-3, whole article.

    PYTHONPATH=src python scripts/analysis/extractive_oracle.py data/runs/<run to compare with>   (box)
"""
import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import polars as pl

from corpus.gold import GoldSet
from evaluate.metrics import normalize_text, rouge_l_score

KH = re.compile(r"kho[aả]n\s+(\d+)\s+Điều\s+(\d+)", re.I)


def clauses(t):
    body = t.split("\n", 1)[1] if "\n" in t else t
    out = {}
    for p in re.split(r"\n(?=\d+\.\s)", body):
        m = re.match(r"(\d+)\.\s", p)
        if m:
            out[m.group(1)] = p
    return out


def main():
    gold = GoldSet().load()
    qa = {str(r["qid"]): r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
    A, T = {}, {}
    for line in open("data/cpt/articles.jsonl", encoding="utf-8"):
        r = json.loads(line)
        k, a = r["key"].split("#", 1)
        A[(k.upper(), a.lower())], T[(k.upper(), a.lower())] = r["text"], r["title"]
    df, N = collections.Counter(), 0
    for t in A.values():
        for c in clauses(t).values():
            N += 1
            df.update(set(normalize_text(c).split()))
    idf = lambda w: math.log((N + 1) / (df.get(w, 0) + 1))   # noqa: E731

    def sc(q, c):
        qw, cw = set(normalize_text(q).split()), collections.Counter(normalize_text(c).split())
        return sum(idf(w) * (1 + math.log(cw[w])) for w in qw if w in cw) / (1 + math.log(1 + sum(cw.values())))
    dev = set(map(str, json.load(open("data/processed/split.json"))["dev"]))
    run = Path(sys.argv[1])
    S = {str(r["qid"]): r["rouge_l"] for r in json.load(open(run / "metrics.json"))["rows"]}
    res, model = collections.defaultdict(list), []
    for qid in dev:
        q = qa.get(qid)
        g = gold.get(int(qid)) if qid.isdigit() else None
        if not q or not g or not g.pairs:
            continue
        ref = q["answer"] or ""
        m = KH.search(ref)
        if not m:
            continue
        kh, dieu = m.group(1), m.group(2)
        art = next(((p.rsplit("#", 1)[0].upper(), dieu) for p in g.pairs if p.rsplit("#", 1)[1] == dieu), None)
        if not art or art not in A:
            continue
        cl = clauses(A[art])
        if kh not in cl or len(cl) < 2:
            continue
        head, law = A[art].split("\n", 1)[0], T[art]
        label = head.split(".")[0]

        def build(ks):
            return f"Căn cứ theo khoản {ks[0]} {label} {law} quy định như sau:\n{head}\n" + "\n".join(cl[k] for k in ks)
        order = sorted(cl, key=lambda k: -sc(q["question"], cl[k]))
        res["oracle clause"].append(rouge_l_score(build([kh]), ref))
        res["selector top-1"].append(rouge_l_score(build(order[:1]), ref))
        res["selector top-1+2"].append(rouge_l_score(build(order[:2]), ref))
        res["selector top-3"].append(rouge_l_score(build(order[:3]), ref))
        res["whole article"].append(rouge_l_score(f"Căn cứ theo {label} {law} quy định như sau:\n" + A[art], ref))
        model.append(S[qid])
    print(f"n = {len(model)} dev questions whose reference names a khoản of a multi-clause gold article")
    print(f"{run.name} (model) mean rouge_l on the same questions: {st.mean(model):.3f}")
    for k, v in res.items():
        print(f"{k:18s} {st.mean(v):.3f}")


if __name__ == "__main__":
    main()
