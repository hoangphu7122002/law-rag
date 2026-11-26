"""Feedback set for the weight-space agent: train prompts (never dev/test) with mean sample ROUGE-L < .65 (tail/mid), shuffled, n rows, decontaminated.
    PYTHONPATH=src python scripts/train/build_wsp_feedback.py data/train/pref_Vp2/cands.jsonl data/wsp/feedback.jsonl --n 120"""
import argparse
import json
import random
import re
import statistics

import polars as pl

from evaluate.metrics import rouge_l_score

norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ap = argparse.ArgumentParser()
ap.add_argument("cands")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=120)
a = ap.parse_args()
sp = json.load(open("data/processed/split.json"))
held = set(sp["test"]) | set(sp["dev"])
qa = pl.read_parquet("data/processed/qa.parquet")
hq = {norm(r["question"]) for r in qa.iter_rows(named=True) if r["qid"] in held}
rows = []
for line in open(a.cands, encoding="utf-8"):
    r = json.loads(line)
    m = re.search(r"### Câu hỏi :\s*(.*?)\s*### Trả lời", r["prompt"], re.S)
    if r["qid"] in held or (m and norm(m.group(1)) in hq):
        continue
    if statistics.mean(rouge_l_score(c, r["reference"]) for c in r["cands"]) < 0.65:
        rows.append({"qid": r["qid"], "prompt": r["prompt"], "reference": r["reference"]})
random.Random(0).shuffle(rows)
import pathlib
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
open(a.out, "w", encoding="utf-8").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows[:a.n])
print(len(rows), "tail/mid candidates;", min(a.n, len(rows)), "written")
