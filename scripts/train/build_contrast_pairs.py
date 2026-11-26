"""Standard pairs + cite-contrast pairs (right-Điều vs wrong-Điều samples of the same prompt), shuffled with a fixed seed.
    PYTHONPATH=src python scripts/train/build_contrast_pairs.py data/train/pref_Vp2/cands.jsonl data/train/pref_Vp2/pairs.jsonl data/train/pref_Vp2/pairs_contrast.jsonl"""
import json
import random
import sys
from collections import Counter

from train.preference.pref_data import contrast_pair

cands, std, out = sys.argv[1:4]
standard = [dict(json.loads(line), source="standard") for line in open(std, encoding="utf-8")]
extra = [p for p in (contrast_pair(json.loads(line)) for line in open(cands, encoding="utf-8")) if p]
mixed = standard + extra
random.Random(0).shuffle(mixed)
open(out, "w", encoding="utf-8").writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in mixed)
print(json.dumps(dict(Counter(p["source"] for p in mixed))))
