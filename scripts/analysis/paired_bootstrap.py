"""Paired bootstrap of the ROUGE-L difference between runs on the same questions (the dev/test sets are small: SE of a mean ~ .015, paired ~ .010).

    PYTHONPATH=src python scripts/analysis/paired_bootstrap.py REF=data/runs/<ref run> [REF2=...] -- NAME=data/runs/<run> ...   (box)
Pools every pair of runs given as `label=dev_run+test_run` (dev and test question ids are disjoint) and prints mean diff vs the reference with a 95% interval.
"""
import json
import random
import sys
from pathlib import Path


def paired_bootstrap(a: dict, b: dict, n: int = 5000, seed: int = 0):
    """Questions scored by both; -> (mean(a - b), 2.5% quantile, 97.5% quantile) of the bootstrap distribution of the mean difference."""
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(n))
    return mean, boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def rows(run_dirs: str) -> dict:
    out = {}
    for d in run_dirs.split("+"):
        for r in json.load(open(Path(d) / "metrics.json"))["rows"]:
            out[(Path(d).name, r["qid"])] = r["rouge_l"]
    return {k[1]: v for k, v in out.items()}   # dev and test qids are disjoint


def main():
    args = [x for x in sys.argv[1:] if "=" in x]
    ref_label, ref_dirs = args[0].split("=", 1)
    ref = rows(ref_dirs)
    print(f"reference {ref_label}: n={len(ref)} mean rouge_l {sum(ref.values()) / len(ref):.4f}")
    print("| model | n | mean | diff vs ref | 95% CI |\n|---|---|---|---|---|")
    for x in args[1:]:
        label, dirs = x.split("=", 1)
        r = rows(dirs)
        d, lo, hi = paired_bootstrap(r, ref)
        print(f"| {label} | {len(set(r) & set(ref))} | {sum(r.values()) / len(r):.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")


if __name__ == "__main__":
    main()
