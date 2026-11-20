"""Automatic checks that do not need a human: does an answer cite what the reference cites, and are its numbers and wording grounded in the retrieved text?

These do NOT measure legal correctness or whether a law is still in force; they measure citation agreement with the reference and faithfulness to
the retrieved passages (invented figures, invented quotations).
"""
import re

from evaluate.metrics import normalize_text

DIEU = re.compile(r"điều\s+(\d+)", re.I)
LAW = re.compile(r"\b\d+/\d{4}/[\wĐđ\-]+", re.I)
NUM = re.compile(r"\d[\d.,]*\d|\d")


def dieu_numbers(text: str) -> set[int]:
    return {int(n) for n in DIEU.findall(text)}


def law_ids(text: str) -> set[str]:
    return {m.lower() for m in LAW.findall(text)}


def citation_match(answer: str, reference: str) -> bool | None:
    """Every article number the reference cites also appears in the answer (None when the reference cites no article)."""
    want = dieu_numbers(reference)
    return None if not want else want <= dieu_numbers(answer)


def law_match(answer: str, reference: str) -> bool | None:
    """The answer names at least one of the legal documents the reference names (None when the reference names none)."""
    want = law_ids(reference)
    return None if not want else bool(want & law_ids(answer))


def numbers(text: str) -> set[str]:
    """Numeric tokens without thousands separators: '1.000.000' -> '1000000'."""
    return {re.sub(r"[.,]", "", n) for n in NUM.findall(text)}


def number_grounding(answer: str, passages: list[str], question: str = "") -> float | None:
    """Share of the answer's numbers that occur in the retrieved passages or in the question (None when the answer has no number)."""
    ans = numbers(answer)
    if not ans:
        return None
    known = numbers(" ".join(passages)) | numbers(question)
    return len(ans & known) / len(ans)


def _grams(text: str, n: int) -> set[tuple]:
    w = normalize_text(text).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def ngram_fidelity(answer: str, passages: list[str], n: int = 8) -> float | None:
    """Share of the answer's word n-grams found verbatim in the passages (high = quotes the sources; None when the answer is shorter than n words)."""
    grams = _grams(answer, n)
    if not grams:
        return None
    src = set().union(*(_grams(p, n) for p in passages)) if passages else set()
    return len(grams & src) / len(grams)


def pick_by_quantile(rows: list[dict], n: int, key: str = "rouge") -> list[dict]:
    """n rows spread evenly over the score range, ordered from the highest score (easy) to the lowest (hard)."""
    ordered = sorted(rows, key=lambda r: r[key], reverse=True)
    if n >= len(ordered):
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (n - 1))] if n > 1 else ordered[0] for i in range(n)]


def sample_rows(rows: list[dict], n: int, seed: int = 7, key: str = "rouge") -> list[dict]:
    """n rows, one from each of n equal-size score strata (random inside the stratum, seeded): a fair sample for a manual check."""
    import random

    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda r: r[key])
    size = len(ordered) / n
    return [rng.choice(ordered[int(i * size):max(int((i + 1) * size), int(i * size) + 1)]) for i in range(n)]


def latex_escape(text: str) -> str:
    out = text.replace("\\", r"\textbackslash{}")
    for ch, rep in {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.items():
        out = out.replace(ch, rep)
    return out
