"""Feedback set for the weight-space agent (train prompts only, decontaminated vs dev/test): n tail/mid rows (mean sample ROUGE-L < .65; rows 0..n-1, field bucket=tailmid) followed by n_head head GUARD rows (>= .65; bucket=head) the edit must not hurt.
    PYTHONPATH=src python scripts/train/build_wsp_feedback.py data/train/pref_Vp2/cands.jsonl data/wsp/feedback.jsonl --n 300 --n-head 100"""
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
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--n-head", type=int, default=100)
a = ap.parse_args()
sp = json.load(open("data/processed/split.json"))
held = set(sp["test"]) | set(sp["dev"])
qa = pl.read_parquet("data/processed/qa.parquet")
hq = {norm(r["question"]) for r in qa.iter_rows(named=True) if r["qid"] in held}
rows, head = [], []
for line in open(a.cands, encoding="utf-8"):
    r = json.loads(line)
    m = re.search(r"### Câu hỏi :\s*(.*?)\s*### Trả lời", r["prompt"], re.S)
    if r["qid"] in held or (m and norm(m.group(1)) in hq):
        continue
    low = statistics.mean(rouge_l_score(c, r["reference"]) for c in r["cands"]) < 0.65
    (rows if low else head).append({"qid": r["qid"], "prompt": r["prompt"], "reference": r["reference"], "bucket": "tailmid" if low else "head"})
random.Random(0).shuffle(rows)
random.Random(1).shuffle(head)
import pathlib
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out = rows[:a.n] + head[:a.n_head]
open(a.out, "w", encoding="utf-8").writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
print(len(rows), "tail/mid and", len(head), "head candidates;", len(out), "written")
