"""Context mixing for SFT rows (docs/research/tail-mid-plan.md): pure helpers on lists of passages.

missing_gold  one pool passage per gold article that the context lacks
inject_all    put ALL of them into the context (rank 1 kept), not just one
maybe_shuffle shuffle the passage order for a fraction of the rows (the model reads the order: reversed -.049, shuffled -.018)
drop_gold     replace gold passages by spare ones (a retrieval miss, the reference answer stays)
"""


def missing_gold(ctx, pool, key_of, gold_keys) -> list:
    have = {key_of(p) for p in ctx}
    out, seen = [], set()
    for p in pool:
        k = key_of(p)
        if k in gold_keys and k not in have and k not in seen:
            out.append(p)
            seen.add(k)
    return out


def inject_all(ctx, extra, rng) -> list:
    out = list(ctx)
    if not extra or not out:
        return out
    slots = list(range(1, len(out))) or [0]
    rng.shuffle(slots)
    for slot, p in zip(slots, extra):
        out[slot] = p
    return out


def maybe_shuffle(ctx, p: float, rng) -> list:
    out = list(ctx)
    if p > 0 and rng.random() < p:
        before = list(out)
        for _ in range(5):   # a permutation equal to the input is not a shuffle
            rng.shuffle(out)
            if out != before or len(out) < 2:
                break
    return out


def drop_gold(ctx, is_gold, spares, rng) -> list:
    spares, out = list(spares), []
    for p in ctx:
        if not is_gold(p):
            out.append(p)
        elif spares:
            out.append(spares.pop(0))
    return out or list(ctx)[:1]
