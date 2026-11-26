"""Score Haiku answers against gold: python scripts/train/score_haiku.py IN_DIR OUT_DIR [OUT_PAIRS]. Keeps answers with rougeL(gold) >= .6, length .75-1.3x gold and rougeL(rejected) lower by >= .1."""
import glob, json, os, sys
from evaluate.metrics import rouge_l_score
ind, outd = sys.argv[1:3]
pairs, tot, kept = [], 0, 0
for f in sorted(glob.glob(f"{outd}/batch_*.jsonl")):
    src = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{ind}/{os.path.basename(f)}", encoding="utf-8")}
    for l in open(f, encoding="utf-8"):
        o = json.loads(l); r = src[o["qid"]]; tot += 1
        a = o["answer"]
        if not a:
            continue
        s, sr, ln = rouge_l_score(a, r["reference"]), rouge_l_score(r["rejected"], r["reference"]), len(a) / len(r["reference"])
        print(os.path.basename(f), o["qid"], round(s, 3), round(sr, 3), round(ln, 2))
        if s >= 0.6 and 0.75 <= ln <= 1.3 and s - sr >= 0.1:
            kept += 1
            pairs.append({"qid": o["qid"], "prompt": r["prompt"], "chosen": a, "rejected": r["rejected"], "margin": round(s - sr, 4), "source": "haiku"})
print({"answers": tot, "kept": kept})
if len(sys.argv) > 3:
    open(sys.argv[3], "w", encoding="utf-8").writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
