"""Preference pairs for DPO from the model's own samples: per train prompt, chosen = the candidate with the highest reward, rejected =
the lowest, same reward as GRPO (src/train/grpo.py `score`: rougeL / overlap / citation, copy and length penalties).

    rows: {"qid", "prompt", "reference", "cands": [str, ...]}   ->   pairs: {"qid", "prompt", "chosen", "rejected", "margin"}
Rows whose reference is a refusal are skipped (the SFT refusal rows already teach that); pairs closer than `min_margin` carry no signal.
"""
from train.grpo import is_refusal, score


def build_pairs(rows: list[dict], min_margin: float = 0.1) -> list[dict]:
    pairs = []
    for r in rows:
        if is_refusal(r["reference"]):
            continue
        scored = sorted(((score(c, r["reference"], r["prompt"]), c) for c in set(r["cands"])), reverse=True)
        (hi, best), (lo, worst) = scored[0], scored[-1]
        if best.strip() and hi - lo >= min_margin:
            pairs.append({"qid": r["qid"], "prompt": r["prompt"], "chosen": best, "rejected": worst, "margin": round(hi - lo, 4)})
    return pairs
