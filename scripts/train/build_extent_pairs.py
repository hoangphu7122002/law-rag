"""Round-2 pairs with balanced length direction: a random subsample of the standard pairs + real extent pairs from the samples + synthetic truncation / extension
negatives, shuffled (fixed seed). See train.preference.pref_data (extent_pairs, truncate_negative, extend_negative, mix_pairs).

    PYTHONPATH=src python scripts/train/build_extent_pairs.py --cands data/train/pref_Vp2/cands.jsonl --standard data/train/pref_Vp2/pairs.jsonl \
        --out data/train/pref_Vp2/pairs_mix.jsonl --keep-standard 700 --n-trunc 150 --n-extend 150
"""
import argparse
import json
import random
from collections import Counter

from train.preference.pref_data import extend_negative, extent_pairs, mix_pairs, truncate_negative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--standard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-standard", type=int, default=700)
    ap.add_argument("--n-trunc", type=int, default=150, help="synthetic under-quoting negatives (the chosen answer cut at a line boundary)")
    ap.add_argument("--n-extend", type=int, default=150, help="synthetic over-quoting negatives (the chosen answer + a context paragraph)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", action="store_true", help="scale to the size of the standard set (final run): keep 67%% of it, n-trunc = n-extend = 14.4%% of it")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(line) for line in open(a.cands, encoding="utf-8")]
    standard = [json.loads(line) for line in open(a.standard, encoding="utf-8")]
    if a.frac:
        a.keep_standard, a.n_trunc, a.n_extend = round(0.67 * len(standard)), round(0.144 * len(standard)), round(0.144 * len(standard))
    extra = [p for r in rows for p in extent_pairs(r)]
    # synthetic negatives hang off the chosen answers of the standard pairs (good answers by construction)
    prompts = {r["qid"]: r["prompt"] for r in rows}
    pool = standard[:]
    rng.shuffle(pool)
    n_t = n_e = 0
    for p in pool:
        if n_t < a.n_trunc:
            neg = truncate_negative(p["chosen"], rng)
            if neg:
                extra.append({"qid": p["qid"], "prompt": p["prompt"], "chosen": p["chosen"], "rejected": neg, "margin": 0.3, "source": "synth_short"})
                n_t += 1
                continue
        if n_e < a.n_extend:
            neg = extend_negative(p["chosen"], prompts.get(p["qid"], p["prompt"]), rng)
            if neg:
                extra.append({"qid": p["qid"], "prompt": p["prompt"], "chosen": p["chosen"], "rejected": neg, "margin": 0.3, "source": "synth_long"})
                n_e += 1
    mixed = mix_pairs(standard, extra, a.keep_standard, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in mixed)
    print(json.dumps({"pairs": len(mixed), "by_source": dict(Counter(p["source"] for p in mixed))}))


if __name__ == "__main__":
    main()
