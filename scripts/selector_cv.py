"""Learned candidate selector, cross-validated on dev: candidates per question = the model answer + "intro + verbatim article" of each of the 5
retrieved passages; a regressor predicts each candidate's ROUGE-L from cheap features, the best predicted one is picked.

    PYTHONPATH=src python scripts/selector_cv.py data/runs/<dev run> [--folds 5]
Exploration on dev only (300 questions: a small, noisy training set), nothing here touches test.
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

from evaluate.faithfulness import dieu_numbers
from evaluate.metrics import rouge_l_score as R

HEADING = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zđ]?)\s*[.:]")


def candidate_features(kind: str, rank: int, score: float, cand: str, model_answer: str, article: str | None) -> list[float]:
    cited = {str(n) for n in dieu_numbers(model_answer)}
    return [1.0 if kind == "model" else 0.0, float(rank), float(score), R(cand, model_answer),
            1.0 if (kind == "model" or (article is not None and article in cited)) else 0.0,
            len(cand) / max(1, len(model_answer))]


def pick_by_prediction(preds: dict, truth: dict) -> float:
    return st.mean(truth[q][max(range(len(p)), key=p.__getitem__)] for q, p in preds.items())


def build(run: Path):
    from corpus import QASet
    from evaluate import RetrievalEval

    qs = {q.qid: q for q in QASet().load("dev")}
    cache, _ = RetrievalEval.cache_of(run)
    psg = {json.loads(l)["qid"]: json.loads(l)["passages"] for l in cache.read_text(encoding="utf-8").splitlines()}
    X, y, qid_of = [], [], []
    for line in open(run / "predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        a, ref = p["answer"], qs[p["qid"]].answer
        head = a.split(":")[0] + ":" if ":" in a else "Căn cứ quy định sau đây:"
        rows = [("model", 0, 0.0, a, None)]
        for i, x in enumerate(psg.get(p["qid"], [])[:5], 1):
            m = HEADING.search(x["text"][:400])
            rows.append(("quote", i, float(x.get("score", 0.0)), head + " " + x["text"].strip(), m.group(1) if m else None))
        for kind, rank, score, cand, art in rows:
            X.append(candidate_features(kind, rank, score, cand, a, art))
            y.append(R(cand, ref))
            qid_of.append(p["qid"])
    return X, y, qid_of


def main() -> None:
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    X, y, qid_of = build(Path(a.run))
    X, y, qid_of = np.array(X), np.array(y), np.array(qid_of)
    qids = sorted(set(qid_of))
    fold = {q: i % a.folds for i, q in enumerate(qids)}
    preds, truth = {}, {}
    for f in range(a.folds):
        te = np.array([fold[q] == f for q in qid_of])
        m = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0).fit(X[~te], y[~te])
        pr = m.predict(X[te])
        for q in sorted(set(qid_of[te])):
            idx = [i for i, qq in enumerate(qid_of[te]) if qq == q]
            preds[q], truth[q] = [pr[i] for i in idx], [y[te][i] for i in idx]
    base = st.mean(t[0] for t in truth.values())            # candidate 0 = the model answer
    sel = pick_by_prediction(preds, truth)
    orc = st.mean(max(t) for t in truth.values())
    print(json.dumps({"questions": len(truth), "model_answer": round(base, 4), "selector_cv": round(sel, 4), "oracle": round(orc, 4),
                      "picked_model_share": round(st.mean(1.0 if max(range(len(p)), key=p.__getitem__) == 0 else 0.0 for p in preds.values()), 3)}))


if __name__ == "__main__":
    main()
