"""A CSV for a human check of answers: 20 test answers, one from each ROUGE-L stratum, with the columns to fill in.

    PYTHONPATH=src python scripts/manual_check_sheet.py <run_dir_name> docs/report/manual_check.csv [--n 20]
Columns to fill (0/1 or short text): right_article, right_content, invented_facts, still_in_force, notes.
"""
import argparse
import csv
import json
from pathlib import Path

from config import load
from corpus import QASet
from evaluate.faithfulness import sample_rows
from evaluate.metrics import rouge_l_score
from retrieve import Retrieval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    cfg = load("configs/exp/A5_v3b.yaml", [f"split={a.split}"])
    qs = {q.qid: q for q in QASet().load(a.split)}
    passages = Retrieval(cfg).run(list(qs.values()))
    preds = {p["qid"]: p["answer"] for p in map(json.loads, open(Path("data/runs") / a.run / "predictions.jsonl", encoding="utf-8"))}
    rows = [{"qid": i, "rouge": rouge_l_score(t, qs[i].answer), "answer": t} for i, t in preds.items()]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "rouge_l", "question", "ground_truth", "system_answer", "retrieved_docs", "right_article", "right_content", "invented_facts", "still_in_force", "notes"])
        for r in sample_rows(rows, a.n):
            q = qs[r["qid"]]
            w.writerow([r["qid"], round(r["rouge"], 3), q.text, q.answer, r["answer"], "; ".join(p.doc for p in passages[r["qid"]]), "", "", "", "", ""])


if __name__ == "__main__":
    main()
