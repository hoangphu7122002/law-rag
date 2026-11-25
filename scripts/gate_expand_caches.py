"""Build retrieval-cache files for the gate + widen experiment (no index, no GPU), so the normal eval path runs it with the retrieval step cached.

    PYTHONPATH=src uv run --with scikit-learn python scripts/gate_expand_caches.py --train data/cache/retrieval/<k10 train> --dev <k20 dev> --test <k20 test>

Gate: gradient boosting on retrieval.gate.features, fitted on TRAIN only; threshold = the 80th percentile of 5-fold out-of-fold train probabilities (flags ~20 %).
Variants (retrieve.k marker -> cache key): 101 base top-5 | 102 gate -> top-10 | 103 gate -> top-15 | 104 always top-15 | 105 oracle (widen to 15 exactly when the
gold article is missing from the top-5; an upper bound, uses gold).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from config import load
from evaluate.retrieval_eval import RetrievalEval
from retrieve.gate import expand_context, features, select_threshold

SHARE = 0.20
qa = {r["qid"]: r for r in pl.read_parquet("data/processed/qa.parquet").iter_rows(named=True)}
gold = {r["qid"]: set(r["pairs"] or ()) for r in pl.read_parquet("data/processed/gold.parquet").iter_rows(named=True)}


def read(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["qid"]] = r["passages"]
    return out


def has_gold(qid, ps):
    keys = {f"{law}#{art}" for law, art in (RetrievalEval.passage_keys(p) for p in ps) if art}
    return bool(gold.get(qid) and gold[qid] & keys)


def xy(cache):
    X, y = [], []
    for q, ps in cache.items():
        if q in qa and gold.get(q):
            X.append(features(qa[q]["question"], ps))
            y.append(0 if has_gold(q, ps[:5]) else 1)
    return np.array(X), np.array(y)


def write(split, marker, rows):
    cfg = load("configs/exp/A5_v3b.yaml", [f"split={split}", f"retrieve.k={marker}"])
    path = Path("data/cache/retrieval") / f"{cfg.retrieval_key()}.jsonl"
    path.write_text("".join(json.dumps({"qid": q, "passages": ps}, ensure_ascii=False) + "\n" for q, ps in rows.items()), encoding="utf-8")
    print(f"  k={marker}: {len(rows)} questions, mean passages {np.mean([len(p) for p in rows.values()]):.2f} -> {path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    Xt, yt = xy(read(a.train))
    gbm = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
    oof = cross_val_predict(gbm, Xt, yt, cv=5, method="predict_proba")[:, 1]
    tau = select_threshold(list(oof), SHARE)
    print(f"train n={len(yt)} miss rate {yt.mean():.3f} OOF AUROC {roc_auc_score(yt, oof):.3f} threshold {tau:.3f} (flags {SHARE:.0%} of train)")
    gbm.fit(Xt, yt)
    for split, path in (("dev", a.dev), ("test", a.test)):
        cache = read(path)
        p = {q: float(gbm.predict_proba(np.array([features(qa[q]["question"], ps)]))[0, 1]) for q, ps in cache.items()}
        flagged = {q: p[q] >= tau for q in cache}
        missed = {q: bool(gold.get(q)) and not has_gold(q, ps[:5]) for q, ps in cache.items()}
        n_m, n_f = sum(missed.values()), sum(flagged.values())
        hit = sum(1 for q in cache if flagged[q] and missed[q])
        print(f"{split}: {len(cache)} questions, flagged {n_f} ({n_f / len(cache):.0%}), misses {n_m}, misses flagged {hit}, precision {hit / max(n_f, 1):.2f}")
        variants = {101: lambda q, ps: expand_context(ps, False), 102: lambda q, ps: expand_context(ps, flagged[q], 5, 10),
                    103: lambda q, ps: expand_context(ps, flagged[q], 5, 15), 104: lambda q, ps: expand_context(ps, True, 5, 15),
                    105: lambda q, ps: expand_context(ps, missed[q], 5, 15)}
        for marker, fn in variants.items():
            rows = {q: fn(q, ps) for q, ps in cache.items()}
            wg = [has_gold(q, rows[q]) for q in cache if gold.get(q)]
            print(f"  gold in the final context: {np.mean(wg):.3f}", end="")
            write(split, marker, rows)


if __name__ == "__main__":
    main()
