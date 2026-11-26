"""Is part of the "cites another Điều" loss a law-VERSION problem (same provision in an older/newer or amending text)?
For each dev question the passage the answer quotes (best token overlap with the answer) is compared with the gold (law, Điều):
  right source | wrong Điều, same law | wrong law, version twin (its text is a near-duplicate of the in-context gold passage) | wrong law, other | no gold passage in context
Mean ROUGE-L of the run's answers per class.
    PYTHONPATH=src python scripts/analysis/diag_version.py data/runs/A5_v3b__2e66c86a   (box)"""
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from config import load
from evaluate.metrics import rouge_l_score
from evaluate.retrieval_eval import RetrievalEval
from retrieve.retrieval import Retrieval

run = Path(sys.argv[1])
cfg = load("configs/exp/A5_v3b.yaml")
final = Retrieval(cfg)._read()
ev = RetrievalEval()
qa = {r["qid"]: r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
dev = set(json.load(open("data/processed/split.json"))["dev"])
answers = {json.loads(l)["qid"]: json.loads(l)["answer"] for l in open(run / "predictions.jsonl", encoding="utf-8")}
tok = lambda s: re.findall(r"\w+", s.lower())
shingles = lambda s, n=5: {tuple(t[i:i + n]) for t in [tok(s)] for i in range(max(len(t) - n + 1, 1))}


def jac(a: str, b: str) -> float:
    x, y = shingles(a), shingles(b)
    return len(x & y) / max(len(x | y), 1)


cls, rl = {}, defaultdict(list)
for qid in sorted(dev & final.keys() & answers.keys()):
    g = ev.gold.get(qid)
    if not (g and g.pairs) or answers[qid].startswith("Các ngữ cảnh"):
        continue
    ps = final[qid]
    keys = [ev.passage_keys({"doc": p.doc, "text": p.text}) for p in ps]
    is_gold = [bool(a and f"{k}#{a}" in g.pairs) for k, a in keys]
    ans = set(tok(answers[qid]))
    src = max(range(len(ps)), key=lambda i: len(ans & set(tok(ps[i].text))) / max(len(set(tok(ps[i].text))), 1))
    if not any(is_gold):
        c = "no gold passage in context"
    elif is_gold[src]:
        c = "right source"
    elif keys[src][0] in g.docs:
        c = "wrong Điều, same law"
    elif any(g_ and jac(ps[src].text, ps[i].text) >= 0.6 for i, g_ in enumerate(is_gold)):
        c = "wrong law, version twin of the gold passage"
    else:
        c = "wrong law, other"
    cls[qid] = c
    rl[c].append(rouge_l_score(answers[qid], qa[qid]["answer"]))
n = len(cls)
print(f"dev questions with a gold pair and an answer (not a refusal): {n}\n| class | n | share | mean rougeL |\n|---|---|---|---|")
for c, k in Counter(cls.values()).most_common():
    print(f"| {c} | {k} | {k / n:.1%} | {st.mean(rl[c]):.3f} |")
