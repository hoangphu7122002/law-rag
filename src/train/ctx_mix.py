"""Context mixing for SFT rows (docs/research/tail-mid-plan.md): pure helpers on lists of passages.

missing_gold  one pool passage per gold article that the context lacks
inject_all    put ALL of them into the context (rank 1 kept), not just one
maybe_shuffle shuffle the passage order for a fraction of the rows (the model reads the order: reversed -.049, shuffled -.018)
ensure_gold   pull the gold articles the context lacks straight from a lookup (corpus), so a row's answer is always supported
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


def ensure_gold(ctx, gold_keys, key_of, lookup, rng) -> list:
    """Every gold article of the reference that is missing from the context is fetched with lookup(key) -> passage | None
    (the corpus, not only the retrieval candidates) and put in; rank 1 stays; keys the lookup does not know are skipped."""
    have = {key_of(p) for p in ctx}
    extra = [p for k in sorted(gold_keys) if k not in have and (p := lookup(k)) is not None]
    return inject_all(ctx, extra, rng) if extra else list(ctx)


def answerable_context(ctx, gold_keys, key_of, has_gold, lookup, rng):
    """The context of a row whose reference answer must be supported: gold put in from the lookup when missing.
    None when that is impossible or when the row has no known gold article (answerability cannot be verified) - the caller drops the row."""
    if not gold_keys:
        return None
    out = ctx if has_gold(ctx) else ensure_gold(ctx, gold_keys, key_of, lookup, rng)
    return out if has_gold(out) else None


def fit_budget(ctx, is_gold, size, budget):
    """Drop non-gold passages from the tail until the context fits `budget` tokens (size(p) = tokens of one passage).
    Gold passages are never dropped; None when the gold alone exceeds the budget (the caller drops the row)."""
    out = list(ctx)
    while sum(size(p) for p in out) > budget:
        spare = [i for i in range(len(out) - 1, -1, -1) if not is_gold(out[i])]
        if not spare:
            return None
        out.pop(spare[0])
    return out


def full_gold(ctx, gold_keys, key_of, lookup, rng) -> list:
    """The reference quotes whole articles, so every gold article must be in the context IN FULL: a gold passage that is only one chunk of its
    article is replaced by the whole article (lookup(key)), the article's other chunks are dropped, and every gold article still missing takes a
    non-gold slot (never rank 1, never another gold passage); the context grows when there are more missing articles than free slots."""
    out, have = [], set()
    for p in ctx:
        k = key_of(p)
        if k in gold_keys:
            if k in have:
                continue
            have.add(k)
            out.append(lookup(k) or p)
        else:
            out.append(p)
    missing = [w for k in sorted(gold_keys) if k not in have and (w := lookup(k)) is not None]
    slots = [i for i in range(1, len(out)) if key_of(out[i]) not in gold_keys]
    rng.shuffle(slots)
    for slot, w in zip(slots, missing):
        out[slot] = w
    out.extend(missing[len(slots):])
    return out
