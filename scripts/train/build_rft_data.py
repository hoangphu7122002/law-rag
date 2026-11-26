"""Best-of-n rejection-sampling SFT data (expert iteration): per training prompt the model's own best sample (ROUGE-L vs reference, length .8-1.3x, rouge >= min), as prompt/completion rows;
val.jsonl is copied from a reference data dir. Shuffled with a fixed seed.
    PYTHONPATH=src python scripts/train/build_rft_data.py data/train/pref_Vp2/cands.jsonl data/train/q35_C data/train/rft_Vp2 --min-rouge 0.55"""
import argparse
import json
import random
import shutil
from pathlib import Path

from train.preference.pref_data import best_of_n_row

ap = argparse.ArgumentParser()
ap.add_argument("cands")
ap.add_argument("valdir")
ap.add_argument("out")
ap.add_argument("--min-rouge", type=float, default=0.55)
a = ap.parse_args()
rows = [r for r in (best_of_n_row(json.loads(line), a.min_rouge) for line in open(a.cands, encoding="utf-8")) if r]
random.Random(0).shuffle(rows)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
open(out / "train.jsonl", "w", encoding="utf-8").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
shutil.copy(Path(a.valdir) / "val.jsonl", out / "val.jsonl")
print(len(rows), "rows")
