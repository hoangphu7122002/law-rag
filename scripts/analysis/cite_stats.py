"""How much does a wrong cited Dieu cost? Splits rows by whether the prediction cites the same "Điều N" as the reference.

    PYTHONPATH=src python scripts/analysis/cite_stats.py <run_dir> [<run_dir> ...]     (box: needs data/runs/<run>/predictions.jsonl + the dev questions)
rows: {"ref", "pred"}; reference rows without any "Điều N" are ignored.
"""
import json
import statistics as st
import sys

from evaluate.metrics import rouge_l_score
from train.grpo import DIEU


def cite_stats(rows: list[dict]) -> dict:
    hit, miss = [], []
    for r in rows:
        want = set(DIEU.findall(r["ref"]))
        if not want:
            continue
        (hit if want <= set(DIEU.findall(r["pred"])) else miss).append(rouge_l_score(r["pred"], r["ref"]))
    n = len(hit) + len(miss)
    mean = lambda x: round(st.mean(x), 4) if x else None  # noqa: E731
    return {"n": n, "match_share": round(len(hit) / n, 3) if n else None, "rouge_match": mean(hit), "rouge_mismatch": mean(miss)}


if __name__ == "__main__":
    from corpus import QASet

    ref = {q.qid: q.answer for q in QASet().load("dev")}
    for run in sys.argv[1:]:
        preds = [json.loads(l) for l in open(f"data/runs/{run}/predictions.jsonl")]
        print(run, cite_stats([{"ref": ref[p["qid"]], "pred": p["answer"]} for p in preds if p["qid"] in ref]))
