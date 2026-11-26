"""CSV of the low-scoring questions of a run, one row per question, for reading by hand.

    PYTHONPATH=src python scripts/analysis/tail_cases.py data/runs/<dev run> --out tail_cases_dev.csv [--low 0.4]
Columns: qid, rougeL, stage (low_score_analysis.failure_stage), law_match, lengths, question, gold articles, retrieved articles (rank order), reference, answer.
Dev only: the reference answers of the test split are not printed.
"""
import argparse
import csv
import json
from pathlib import Path

from evaluate.faithfulness import law_ids
from low_score_analysis import failure_stage


def main() -> None:
    from corpus import QASet
    from corpus.gold import GoldSet
    from evaluate import RetrievalEval

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--low", type=float, default=0.4)
    a = ap.parse_args()
    run = Path(a.run)
    qs = {q.qid: q for q in QASet().load("dev")}
    gold = GoldSet().load()
    cache, _ = RetrievalEval.cache_of(run)
    psg = {json.loads(l)["qid"]: json.loads(l)["passages"] for l in cache.read_text(encoding="utf-8").splitlines()}
    sc = {r["qid"]: r["rouge_l"] for r in json.load(open(run / "metrics.json"))["rows"]}
    rows = []
    for line in open(run / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        q = qs.get(p["qid"])
        if q is None or sc[p["qid"]] >= a.low:
            continue
        g = gold.get(p["qid"])
        keys = [RetrievalEval.passage_keys(x) for x in psg.get(p["qid"], [])[:5]]
        pairs = {f"{k}#{art}" for k, art in keys if art}
        ref_laws, ans_laws = law_ids(q.answer), law_ids(p["answer"])
        rows.append({"qid": p["qid"], "rougeL": round(sc[p["qid"]], 3), "stage": failure_stage(set(g.pairs) if g else set(), pairs, p["answer"]),
                     "law_match": "" if not ref_laws else int(bool(ref_laws & ans_laws)), "len_answer": len(p["answer"]), "len_reference": len(q.answer),
                     "question": q.text, "gold_articles": " ; ".join(sorted(g.pairs)) if g else "", "retrieved_top5": " ; ".join(f"{k}#{art or '?'}" for k, art in keys),
                     "reference": q.answer.replace("\n", " "), "answer": p["answer"].replace("\n", " ")})
    rows.sort(key=lambda r: (r["stage"], r["rougeL"]))
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", a.out)


if __name__ == "__main__":
    main()
