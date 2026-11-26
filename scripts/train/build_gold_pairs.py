"""Gold-as-chosen pairs for tail/mid prompts that had no pair (export_haiku_targets batches), added to a pair file and shuffled (fixed seed).
Hard decontamination: drops any row whose qid, normalised question, or normalised gold answer is in dev/test.
    PYTHONPATH=src python scripts/train/build_gold_pairs.py data/train/haiku_targets data/train/pref_Vp2/pairs_mix.jsonl data/train/pref_Vp2/pairs_mix_gold.jsonl"""
import glob
import json
import random
import re
import sys

import polars as pl


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def question_of(prompt):
    m = re.search(r"### Câu hỏi :\s*(.*?)\s*### Trả lời", prompt, re.S)
    return m.group(1) if m else ""


tdir, base, out = sys.argv[1:4]
split = json.load(open("data/processed/split.json"))
held = set(split.get("test", [])) | set(split.get("dev", []))
qa = pl.read_parquet("data/processed/qa.parquet")
hq = {norm(r["question"]) for r in qa.iter_rows(named=True) if r["qid"] in held}
ha = {norm(r["answer"]) for r in qa.iter_rows(named=True) if r["qid"] in held}
pairs = [json.loads(line) for line in open(base, encoding="utf-8")]
n0, new, dropped = len(pairs), [], 0
for f in sorted(glob.glob(f"{tdir}/batch_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["qid"] in held or norm(question_of(r["prompt"])) in hq or norm(r["reference"]) in ha:
            dropped += 1
            continue
        new.append({"qid": r["qid"], "prompt": r["prompt"], "chosen": r["reference"], "rejected": r["rejected"], "margin": 0.3, "source": "gold_tail"})
pairs += new
random.Random(0).shuffle(pairs)
open(out, "w", encoding="utf-8").writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
print(json.dumps({"base": n0, "gold_added": len(new), "dropped_heldout": dropped, "total": len(pairs)}))
