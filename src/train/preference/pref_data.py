"""Preference pairs for DPO from the model's own samples: per train prompt, chosen = the candidate with the highest reward, rejected =
the lowest, same reward as GRPO (src/train/preference/grpo.py `score`: rougeL / overlap / citation, copy and length penalties).

    rows: {"qid", "prompt", "reference", "cands": [str, ...]}   ->   pairs: {"qid", "prompt", "chosen", "rejected", "margin"}
Rows whose reference is a refusal are skipped (the SFT refusal rows already teach that); pairs closer than `min_margin` carry no signal.
"""
import re

from train.preference.grpo import is_refusal, score

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


# ---- extent pairs: balance the length direction of the preference signal (2026-11-26) -------------------------------------------------------------
# Diagnosis: in the standard best-vs-worst pairs the rejected answer is 1.45-1.71x the reference length and the chosen one is shorter than the rejected one by >20 %
# in 45-55 % of the pairs, so DPO mostly learns "shorter" (under-quoting rose 44 -> 52 cases on dev). Add negatives in both directions and a few hard pairs.
from evaluate.metrics import rouge_l_score  # noqa: E402


def truncate_negative(answer: str, rng, lo: float = 0.35, hi: float = 0.6) -> str | None:
    """The answer cut at a line boundary to lo..hi of its length (an under-quoting negative); None when it has too few lines."""
    lines = answer.split("\n")
    if len(lines) < 3:
        return None
    total, acc = len(answer), 0
    target = rng.uniform(lo, hi) * total
    for i, line in enumerate(lines):
        acc += len(line) + 1
        if acc >= target and i >= 1:
            out = "\n".join(lines[: i + 1]).rstrip()
            return out if len(out) < 0.8 * total else None
    return None


def extend_negative(answer: str, prompt: str, rng, min_len: int = 120) -> str | None:
    """The answer followed by a context paragraph it does not contain (an over-quoting negative); None when the prompt has no such paragraph."""
    cands = [p.strip() for p in prompt.split("\n") if len(p.strip()) >= min_len and p.strip()[:60] not in answer]
    return answer.rstrip() + "\n" + rng.choice(cands) if cands else None


def extent_pairs(row: dict, good_rouge: float = 0.65, bad_rouge: float = 0.55) -> list[dict]:
    """Real pairs from the model's own samples: a good answer (rouge >= good_rouge, length 0.75-1.3x the reference) against a short bad one (< 0.7x), a long bad one
    (> 1.4x) and a same-length mid-quality one (rouge .35-.6, length .7-1.4x). Up to one pair of each kind per prompt."""
    ref = row["reference"]
    if is_refusal(ref):
        return []
    n = max(len(ref), 1)
    scored = [(rouge_l_score(c, ref), len(c) / n, c) for c in dict.fromkeys(row["cands"])]
    good = sorted((s for s in scored if s[0] >= good_rouge and 0.75 <= s[1] <= 1.3), key=lambda s: -s[0])
    if not good:
        return []
    best = good[0]
    pools = {"real_short": [s for s in scored if s[1] < 0.7 and s[0] < bad_rouge],
             "real_long": [s for s in scored if s[1] > 1.4 and s[0] < bad_rouge],
             "real_hard": [s for s in scored if 0.35 <= s[0] <= 0.6 and 0.7 <= s[1] <= 1.4 and s[2] != best[2]]}
    out = []
    for kind, pool in pools.items():
        if pool:
            worst = min(pool, key=lambda s: s[0])
            out.append({"qid": row["qid"], "prompt": row["prompt"], "chosen": best[2], "rejected": worst[2], "margin": round(best[0] - worst[0], 4), "source": kind})
    return out


def mix_pairs(standard: list[dict], extra: list[dict], keep_standard: int, seed: int = 0) -> list[dict]:
    """A random `keep_standard` of the standard pairs plus all extra pairs, shuffled (fixed seed); every pair carries its `source`."""
    import random
    rng = random.Random(seed)
    std = [dict(p, source=p.get("source", "standard")) for p in rng.sample(standard, min(keep_standard, len(standard)))]
    out = std + list(extra)
    rng.shuffle(out)
    return out
