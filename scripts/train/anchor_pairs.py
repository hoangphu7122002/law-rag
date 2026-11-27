"""Standard DPO pairs -> anchored pairs (train.preference.pref_data.anchor_pairs): no poor answer is taught as the target.

    PYTHONPATH=src python scripts/train/anchor_pairs.py --cands data/train/pref_v3b/cands.jsonl --pairs data/train/pref_v3b/pairs.jsonl \
        --out data/train/pref_v3b/pairs_anchor_ref.jsonl --mode ref [--min-chosen 0.5]
"""
import argparse
import json
from collections import Counter

from train.preference.pref_data import anchor_pairs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["drop", "ref"], default="ref")
    ap.add_argument("--min-chosen", type=float, default=0.5)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.cands)]
    pairs = [json.loads(l) for l in open(a.pairs)]
    out = anchor_pairs(rows, pairs, a.min_chosen, a.mode)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in out)
    print(json.dumps({"standard_in": len(pairs), "out": len(out), **Counter(p["source"] for p in out)}))
