"""Paired bootstrap: resample the same questions for both runs; report the mean difference and its 95 % interval.

A change is kept only if the interval is above 0 (AGENTS.md tuning rule).
"""
import numpy as np


class PairedBootstrap:
    def __init__(self, n: int = 10_000, seed: int = 0):
        self.n, self.seed = n, seed

    def compare(self, a: dict[int, float], b: dict[int, float]) -> dict:
        qids = sorted(set(a) & set(b))
        diff = np.array([b[q] - a[q] for q in qids])
        rng = np.random.default_rng(self.seed)
        means = diff[rng.integers(0, len(diff), size=(self.n, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return {"n": len(qids), "diff": float(diff.mean()), "lo": float(lo), "hi": float(hi),
                "p_b_not_better": float((means <= 0).mean())}
