"""Automatic checks + worked examples of one run, for the report appendix.

    PYTHONPATH=src python scripts/analysis/appendix_examples.py <run_dir_name> [--split test] [--n 12] [--out docs/report/paper/appendix_examples.tex]
Writes (1) data/runs/<run>/faithfulness.json: citation / law agreement with the reference, number grounding and quotation rate of the answers
(and of the references, as context); (2) a LaTeX file with n examples spread from the easiest to the hardest question by ROUGE-L, each shown with
its ground-truth answer. Retrieval comes from the cache of the run's own configuration.
"""
import argparse
import json
import statistics as st
from pathlib import Path

from config import load
from corpus import QASet
from evaluate import faithfulness as F
from evaluate.metrics import rouge_l_score
from retrieve import Retrieval


def cut(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="docs/report/paper/appendix_examples.tex")
    a = ap.parse_args()
    run = Path("data/runs") / a.run
    cfg = load("configs/exp/A5_v3b.yaml", [f"split={a.split}"])
    qs = {q.qid: q for q in QASet().load(a.split)}
    passages = Retrieval(cfg).run(list(qs.values()))
    preds = {p["qid"]: p["answer"] for p in map(json.loads, open(run / "predictions.jsonl", encoding="utf-8"))}
    rows = []
    for qid, ans in preds.items():
        q, texts = qs[qid], [p.text for p in passages[qid]]
        rows.append({"qid": qid, "question": q.text, "reference": q.answer, "answer": ans, "rouge": rouge_l_score(ans, q.answer),
                     "cite": F.citation_match(ans, q.answer), "law": F.law_match(ans, q.answer),
                     "num": F.number_grounding(ans, texts, q.text), "quote": F.ngram_fidelity(ans, texts),
                     "ref_num": F.number_grounding(q.answer, texts, q.text), "ref_quote": F.ngram_fidelity(q.answer, texts)})
    stats = {"n": len(rows), "rouge_l": mean(r["rouge"] for r in rows),
             "citation_match": rate(r["cite"] for r in rows), "law_match": rate(r["law"] for r in rows),
             "number_grounding": mean(r["num"] for r in rows), "numbers_fully_grounded": rate(r["num"] == 1.0 for r in rows if r["num"] is not None),
             "quote_rate_8gram": mean(r["quote"] for r in rows),
             "reference_number_grounding": mean(r["ref_num"] for r in rows), "reference_quote_rate_8gram": mean(r["ref_quote"] for r in rows)}
    (run / "faithfulness.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))

    def mark(v):
        return "n/a" if v is None else ("yes" if v else "no")

    tex = []
    for i, r in enumerate(F.pick_by_quantile(rows, a.n), 1):
        e = F.latex_escape
        tex.append("\\paragraph{Example %d (qid %d) --- \\rl{} %.3f.} Same \\dieu{} as the reference: %s; same law: %s; numbers grounded in sources: %s.\n"
                   "\\begin{quote}\\small\\textbf{Question.} %s\\\\[2pt]\n\\textbf{Ground truth.} %s\\\\[2pt]\n\\textbf{System answer.} %s\\end{quote}\n"
                   % (i, r["qid"], r["rouge"], mark(r["cite"]), mark(r["law"]), "n/a" if r["num"] is None else "%.0f\\%%" % (100 * r["num"]),
                      e(cut(r["question"], 400)), e(cut(r["reference"], 520)), e(cut(r["answer"], 520))))
    Path(a.out).write_text("\n".join(tex), encoding="utf-8")


if __name__ == "__main__":
    main()
