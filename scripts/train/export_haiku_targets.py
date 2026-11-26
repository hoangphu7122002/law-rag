"""Export train prompts that have NO preference pair and sit in the tail/mid ROUGE-L bucket (mean of the n=8 samples < .65) as small batches for Haiku to answer.
Decontaminated: qid and normalised question text must not appear in dev/test (data/processed/split.json + qa.parquet).
    PYTHONPATH=src python scripts/train/export_haiku_targets.py data/train/pref_Vp2/cands.jsonl data/train/pref_Vp2/pairs_mix.jsonl data/train/haiku_targets --batch 6"""
import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import polars as pl

from evaluate.metrics import rouge_l_score


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def question_of(prompt: str) -> str:
    m = re.search(r"### Câu hỏi :\s*(.*?)\s*### Trả lời", prompt, re.S)
    return m.group(1) if m else ""


ap = argparse.ArgumentParser()
ap.add_argument("cands")
ap.add_argument("pairs")
ap.add_argument("outdir")
ap.add_argument("--batch", type=int, default=6)
ap.add_argument("--max-mean", type=float, default=0.65)
a = ap.parse_args()
split = json.load(open("data/processed/split.json"))
held = set(split.get("test", [])) | set(split.get("dev", []))
qa = pl.read_parquet("data/processed/qa.parquet")
held_q = {norm(r["question"]) for r in qa.iter_rows(named=True) if r["qid"] in held}
paired = {json.loads(line)["qid"] for line in open(a.pairs, encoding="utf-8")}
out, drop = [], Counter()
for line in open(a.cands, encoding="utf-8"):
    r = json.loads(line)
    if r["qid"] in paired:
        continue
    if r["qid"] in held or norm(question_of(r["prompt"])) in held_q:
        drop["held_out"] += 1
        continue
    m = statistics.mean(rouge_l_score(c, r["reference"]) for c in r["cands"])
    if m >= a.max_mean:
        drop["head"] += 1
        continue
    worst = min(r["cands"], key=lambda c: rouge_l_score(c, r["reference"]))
    out.append({"qid": r["qid"], "prompt": r["prompt"], "reference": r["reference"], "rejected": worst, "mean_rouge": round(m, 3)})
Path(a.outdir).mkdir(parents=True, exist_ok=True)
for i in range(0, len(out), a.batch):
    with open(Path(a.outdir) / f"batch_{i // a.batch:03d}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in out[i:i + a.batch])
print(json.dumps({"targets": len(out), "dropped": dict(drop)}))
