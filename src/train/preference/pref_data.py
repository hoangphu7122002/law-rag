"""Preference pairs for DPO from the model's own samples: per train prompt, chosen = the candidate with the highest reward, rejected =
the lowest, same reward as GRPO (src/train/grpo.py `score`: rougeL / overlap / citation, copy and length penalties).

    rows: {"qid", "prompt", "reference", "cands": [str, ...]}   ->   pairs: {"qid", "prompt", "chosen", "rejected", "margin"}
Rows whose reference is a refusal are skipped (the SFT refusal rows already teach that); pairs closer than `min_margin` carry no signal.
"""
import re

from train.grpo import is_refusal, score

KHOAN = re.compile(r"kho[aả]n\s+(\d+)", re.I)


def extent_bonus(cand: str, reference: str, khoan_weight: float, length_weight: float) -> float:
    """Reward for quoting the right extent (diagnosis 2026-11-26: with the right Điều cited, ROUGE-L is .84 when the answer is 0.7-1.3x the reference length
    and .55 otherwise; and the reference's khoản is usually named): + khoan_weight when the reference names no khoản or the candidate names all of them,
    + length_weight when len(candidate) / len(reference) is within 0.7..1.3."""
    ref_k = set(KHOAN.findall(reference))
    bonus = khoan_weight if (not ref_k or ref_k <= set(KHOAN.findall(cand))) else 0.0
    ratio = len(cand) / max(len(reference), 1)
    return bonus + (length_weight if 0.7 <= ratio <= 1.3 else 0.0)


def build_pairs(rows: list[dict], min_margin: float = 0.1, cite_weight: float = 0.2, khoan_weight: float = 0.0, length_weight: float = 0.0) -> list[dict]:
    pairs = []
    for r in rows:
        if is_refusal(r["reference"]):
            continue
        scored = sorted(((score(c, r["reference"], r["prompt"], cite_weight=cite_weight) + (extent_bonus(c, r["reference"], khoan_weight, length_weight) if (khoan_weight or length_weight) else 0.0), c)
                         for c in set(r["cands"])), reverse=True)
        (hi, best), (lo, worst) = scored[0], scored[-1]
        if best.strip() and hi - lo >= min_margin:
            pairs.append({"qid": r["qid"], "prompt": r["prompt"], "chosen": best, "rejected": worst, "margin": round(hi - lo, 4)})
    return pairs
