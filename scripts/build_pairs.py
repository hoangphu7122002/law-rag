"""cands.jsonl -> pairs.jsonl (DPO), parallel over rows (the pure-python ROUGE-L is slow).

    PYTHONPATH=src python scripts/build_pairs.py data/train/pref_v3b/cands.jsonl data/train/pref_v3b/pairs.jsonl [--min-margin 0.1]
"""
import argparse
import json
from functools import partial
from multiprocessing import Pool

from train.pref_data import build_pairs


def one(min_margin: float, cite_weight: float, row: dict) -> list[dict]:
    p = build_pairs([row], min_margin, cite_weight)
    if p:
        p[0]["chosen_is_greedy"] = p[0]["chosen"] == row["cands"][0]
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cands")
    ap.add_argument("out")
    ap.add_argument("--min-margin", type=float, default=0.1)
    ap.add_argument("--cite-weight", type=float, default=0.2, help="weight of the cited-Điều term in the reward (0.2 = the original reward)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.cands)]
    with Pool(24) as pool:
        pairs = [p for ps in pool.map(partial(one, a.min_margin, a.cite_weight), rows, chunksize=8) for p in ps]
    with open(a.out, "w") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
    print(json.dumps({"rows": len(rows), "pairs": len(pairs), "mean_margin": round(sum(p["margin"] for p in pairs) / max(1, len(pairs)), 3),
                      "chosen_is_greedy": round(sum(p["chosen_is_greedy"] for p in pairs) / max(1, len(pairs)), 3)}))
