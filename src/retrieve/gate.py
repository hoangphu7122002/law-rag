"""Retrieval-sufficiency gate (docs/research/): a tiny classical classifier on retrieval statistics says "gold article probably NOT in the top-5";
for flagged questions the context is widened (more reranked passages). No LLM, no GPU. Pure helpers; the classifier is fitted in scripts/analysis/gate_expand_caches.py.

    features(question, passages)        28 numbers: top-10 reranker scores, gaps, slope, spread, how many laws / articles, named-law match
    select_threshold(probs, share)      the probability above which `share` of the questions are flagged
    expand_context(passages, flagged)   top base_k, or top wide_k when flagged
Passages are dicts {doc, text, score} as in the retrieval cache.
"""
import math

from corpus.keys import law_key, stem_key
from evaluate.retrieval_eval import RetrievalEval
from generate.qprofile import question_profile

N = 10


def _skew(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return 0.0 if sd < 1e-9 else sum(((x - m) / sd) ** 3 for x in xs) / len(xs)


def features(question: str, passages: list[dict]) -> list[float]:
    sc = [float(p["score"]) for p in passages[:N]]
    sc += [0.0] * (N - len(sc))
    docs5 = [stem_key(p["doc"]) for p in passages[:5]]
    docs10 = [stem_key(p["doc"]) for p in passages[:N]]
    arts5 = {f"{law}#{art}" for law, art in (RetrievalEval.passage_keys(p) for p in passages[:5]) if art}
    prof = question_profile(question)
    named = {law_key(x) for x in prof.get("laws", [])}
    named_in = (1.0 if named & set(docs5) else 0.0) if named else -1.0
    n = len(sc)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(sc) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, sc)) / sum((x - mx) ** 2 for x in xs)
    return [*sc, sc[0] - sc[1], sc[0] - sc[4], sc[0] - sc[9], slope, math.sqrt(sum((y - my) ** 2 for y in sc) / n), _skew(sc),
            float(len(set(docs5))), float(len(arts5)), float(len(set(docs10))), float(docs5.count(docs5[0])) if docs5 else 0.0,
            float(sum(s > 0.5 for s in sc)), float(sum(s > 0.9 for s in sc)), sum(sc[:5]) / 5 - sum(sc[5:]) / 5,
            float(bool(named)), named_in, float(len(question)), float(bool(prof.get("dieu"))), float(bool(prof.get("years")))]


def select_threshold(probs: list[float], share: float) -> float:
    """Smallest threshold that still flags at least round(share * n) questions (ties flag together)."""
    k = max(1, round(share * len(probs)))
    return sorted(probs, reverse=True)[k - 1]


def expand_context(passages: list, flagged: bool, base_k: int = 5, wide_k: int = 15) -> list:
    return list(passages[: wide_k if flagged else base_k])
