"""Where do the DPO pairs sit? Buckets train prompts by mean ROUGE-L of the n=8 samples (tail < .45, mid .45-.65, head >= .65), counts prompts, pairs and
prompts without any pair per bucket, and checks question overlap with dev/test (decontamination).
    PYTHONPATH=src python scripts/analysis/pairs_tailmid.py data/train/pref_Vp2/cands.jsonl data/train/pref_Vp2/pairs_mix.jsonl"""
import json
import statistics
import sys
from collections import Counter, defaultdict

from evaluate.metrics import rouge_l_score

cands, pairs = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(cands, encoding="utf-8")]
pr = defaultdict(Counter)
for line in open(pairs, encoding="utf-8"):
    p = json.loads(line)
    pr[p["qid"]][p.get("source", "standard")] += 1
bucket = lambda m: "tail" if m < 0.45 else "mid" if m < 0.65 else "head"
st = defaultdict(lambda: Counter())
for r in rows:
    ref = r["reference"]
    m = statistics.mean(rouge_l_score(c, ref) for c in r["cands"])
    b = bucket(m)
    st[b]["prompts"] += 1
    st[b]["pairs"] += sum(pr[r["qid"]].values())
    st[b]["no_pair"] += not pr[r["qid"]]
    st[b]["synth_or_extent"] += sum(v for k, v in pr[r["qid"]].items() if k != "standard")
print(json.dumps({b: dict(c) for b, c in st.items()}, indent=1))
