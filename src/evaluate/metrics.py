"""Answer metrics, a line-for-line port of the course baseline notebook (Baseline.ipynb).

Keep these identical to the baseline so every comparison is fair. Cosine needs an embedding function,
so it is passed in (the baseline uses keepitreal/vietnamese-sbert).
"""
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("\n", " ").strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def cosine_similarity_score(pred: str, true: str, embed: Callable[[str], Sequence[float]]) -> float:
    if not pred or not true:
        return 0.0
    a, b = np.asarray(embed(pred), dtype=float), np.asarray(embed(true), dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def jaccard_similarity(pred: str, true: str) -> float:
    pred_words = set(normalize_text(pred).split())
    true_words = set(normalize_text(true).split())
    return len(pred_words & true_words) / len(pred_words | true_words) if pred_words and true_words else 0.0


def token_overlap_score(pred: str, true: str) -> float:
    pred_words = normalize_text(pred).split()
    true_words = normalize_text(true).split()
    if not true_words:
        return 0.0
    return sum((Counter(pred_words) & Counter(true_words)).values()) / len(true_words)


def bleu_score_simple(pred: str, true: str) -> float:
    pred_words = normalize_text(pred).split()
    true_words = normalize_text(true).split()
    if not pred_words or not true_words:
        return 0.0
    overlap = sum((Counter(pred_words) & Counter(true_words)).values())
    precision = overlap / len(pred_words)
    bp = 1.0 if len(pred_words) >= len(true_words) else float(np.exp(1 - len(true_words) / len(pred_words)))
    return bp * precision


def rouge_l_score(pred: str, true: str) -> float:
    pred_words = normalize_text(pred).split()
    true_words = normalize_text(true).split()
    if not pred_words or not true_words:
        return 0.0
    m, n = len(pred_words), len(true_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == true_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def all_lexical(pred: str, true: str) -> dict[str, float]:
    """Every metric except cosine (which needs a model)."""
    return {
        "jaccard": jaccard_similarity(pred, true),
        "token_overlap": token_overlap_score(pred, true),
        "bleu1": bleu_score_simple(pred, true),
        "rouge_l": rouge_l_score(pred, true),
    }
