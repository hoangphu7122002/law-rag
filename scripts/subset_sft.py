"""A nested, seeded subset of an SFT dataset by question (every row of a kept question stays), for the learning curve.

    python scripts/subset_sft.py data/train/q35_v3b data/train/q35_v3b_f25_s1 --fraction 0.25 --seed 1
val.jsonl is copied unchanged so validation loss is comparable across sizes.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _rank(qid, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def subset_rows(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    if fraction >= 1.0:
        return rows
    qids = sorted({r["qid"] for r in rows}, key=lambda q: _rank(q, seed))      # same order for every fraction -> nested subsets
    keep = set(qids[: round(len(qids) * fraction)])
    return [r for r in rows if r["qid"] in keep]


def write_subset(src: Path, out: Path, fraction: float, seed: int) -> int:
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (src / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sub = subset_rows(rows, fraction, seed)
    (out / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sub) + "\n", encoding="utf-8")
    shutil.copy(src / "val.jsonl", out / "val.jsonl")
    return len(sub)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    print(write_subset(a.src, a.out, a.fraction, a.seed), "rows ->", a.out)
