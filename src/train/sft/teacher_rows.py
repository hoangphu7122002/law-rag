"""Verified teacher answers as extra SFT rows (a second answer per train question).

    PYTHONPATH=src python -m train.sft.teacher_rows --runs data/runs/<run>/predictions.jsonl ... --out data/teacher_rows.jsonl

A run answer is kept when it is GOOD (ROUGE-L >= .6 against the course reference AND it cites a gold Điều, train.react.react_data.grade) and is not a copy
of the reference; across runs the best ROUGE-L per question wins. The teacher never saw the reference (pass 1 = profile + top-5).
"""
import argparse
import json
from pathlib import Path


def select_teacher_rows(runs: list[dict], reference_of, gold_dieu_of, grade, exclude_copy: bool = True) -> list[dict]:
    best: dict = {}
    for run in runs:
        for qid, answer in run.items():
            ref = reference_of(qid)
            g = grade(answer, ref, gold_dieu_of(qid))
            if not g["ok"] or (exclude_copy and (answer.strip() == (ref or "").strip() or g["rouge"] >= 0.98)):
                continue
            if qid not in best or g["rouge"] > best[qid][0]:
                best[qid] = (g["rouge"], answer)
    return [{"qid": q, "answer": a, "rouge": r} for q, (r, a) in sorted(best.items())]


def main() -> None:
    import polars as pl

    from train.react.react_build import gold_dieu
    from train.react.react_data import grade

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="predictions.jsonl of teacher runs on train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {r["qid"]: r["answer"] for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
    gold = {r["qid"]: gold_dieu(r["pairs"]) for r in pl.read_parquet("data/processed/gold.parquet").iter_rows(named=True)}
    runs = []
    for f in a.runs:
        rows = [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
        runs.append({int(r["qid"]): r["answer"] for r in rows if r["qid"] in ref and r.get("answer") and not r.get("error")})
    out = select_teacher_rows(runs, ref.get, lambda q: gold.get(q, set()), grade)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{len(out)} teacher rows from {sum(len(r) for r in runs)} answers -> {a.out}")


if __name__ == "__main__":
    main()
