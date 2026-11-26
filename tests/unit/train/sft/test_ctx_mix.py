import random

from records import Passage
from train.sft.ctx_mix import drop_gold, inject_all, maybe_shuffle, missing_gold


def P(pid):
    return Passage(pid=pid, doc="d", text=pid)


KEY = lambda p: p.pid   # noqa: E731 - the passage id stands for its "<law>#<Dieu>" key


def test_missing_gold_returns_one_pool_passage_per_absent_gold_article():
    ctx = [P("a#1"), P("b#2"), P("c#3")]
    pool = [P("a#1"), P("x#9"), P("g#1"), P("g#1"), P("h#4")]
    got = missing_gold(ctx, pool, KEY, {"a#1", "g#1", "h#4", "z#0"})
    assert [p.pid for p in got] == ["g#1", "h#4"]   # a#1 already there, z#0 not in the pool


def test_inject_all_puts_every_missing_gold_in_and_keeps_rank_one():
    ctx = [P("r1"), P("r2"), P("r3"), P("r4"), P("r5")]
    out = inject_all(ctx, [P("g1"), P("g2")], random.Random(0))
    assert len(out) == 5 and out[0].pid == "r1"
    assert {"g1", "g2"} <= {p.pid for p in out}


def test_inject_all_never_exceeds_the_context_size():
    ctx = [P("r1"), P("r2")]
    out = inject_all(ctx, [P("g1"), P("g2"), P("g3")], random.Random(1))
    assert len(out) == 2 and out[0].pid == "r1" and out[1].pid in {"g1", "g2", "g3"}


def test_inject_all_single_slot_context_is_replaced():
    assert [p.pid for p in inject_all([P("r1")], [P("g1")], random.Random(0))] == ["g1"]


def test_shuffle_probability_zero_and_one():
    ctx = [P(str(i)) for i in range(8)]
    assert maybe_shuffle(ctx, 0.0, random.Random(0)) == ctx
    out = maybe_shuffle(ctx, 1.0, random.Random(0))
    assert sorted(p.pid for p in out) == sorted(p.pid for p in ctx) and out != ctx


def test_shuffle_does_not_mutate_the_input():
    ctx = [P(str(i)) for i in range(6)]
    before = list(ctx)
    maybe_shuffle(ctx, 1.0, random.Random(3))
    assert ctx == before


def test_drop_gold_replaces_gold_with_spares_in_place():
    ctx = [P("a"), P("g"), P("b")]
    out = drop_gold(ctx, lambda p: p.pid == "g", [P("s1"), P("s2")], random.Random(0))
    assert [p.pid for p in out] == ["a", "s1", "b"]


def test_drop_gold_without_spare_removes_but_keeps_one_passage():
    out = drop_gold([P("g1"), P("g2")], lambda p: p.pid.startswith("g"), [], random.Random(0))
    assert len(out) == 1


def test_ensure_gold_pulls_missing_gold_articles_from_a_lookup_and_keeps_rank_one():
    from train.sft.ctx_mix import ensure_gold

    store = {"g#1": P("g#1"), "h#2": P("h#2")}
    ctx = [P("r1"), P("r2"), P("r3"), P("r4")]
    out = ensure_gold(ctx, {"g#1", "h#2", "z#9"}, KEY, store.get, random.Random(0))
    assert out[0].pid == "r1" and len(out) == 4
    assert {"g#1", "h#2"} <= {p.pid for p in out}          # z#9 is not in the store: skipped, not an error


def test_ensure_gold_leaves_a_context_that_already_has_the_gold_alone():
    from train.sft.ctx_mix import ensure_gold

    ctx = [P("r1"), P("g#1")]
    assert ensure_gold(ctx, {"g#1"}, KEY, lambda k: None, random.Random(0)) == ctx


def test_answerable_context_returns_none_when_the_gold_cannot_be_put_in():
    from train.sft.ctx_mix import answerable_context

    has = lambda ps: any(p.pid.startswith("g") for p in ps)   # noqa: E731
    ctx = [P("r1"), P("r2")]
    assert answerable_context(ctx, {"g#1"}, KEY, has, {"g#1": P("g#1")}.get, random.Random(0))[0].pid == "r1"
    assert answerable_context(ctx, {"z#9"}, KEY, has, {}.get, random.Random(0)) is None      # gold not in the corpus lookup
    assert answerable_context(ctx, set(), KEY, has, {}.get, random.Random(0)) is None        # no known gold: answerability unverifiable
    assert answerable_context([P("g#1")], {"g#1"}, KEY, has, {}.get, random.Random(0)) == [P("g#1")]


def test_fit_budget_drops_non_gold_passages_from_the_tail_and_never_the_gold():
    from train.sft.ctx_mix import fit_budget

    ctx = [P("r1"), P("g#1"), P("r3"), P("r4"), P("r5")]
    size = lambda p: 10   # noqa: E731
    is_gold = lambda p: p.pid.startswith("g")   # noqa: E731
    assert [p.pid for p in fit_budget(ctx, is_gold, size, 30)] == ["r1", "g#1", "r3"]
    assert [p.pid for p in fit_budget(ctx, is_gold, size, 100)] == [p.pid for p in ctx]
    assert [p.pid for p in fit_budget(ctx, is_gold, size, 10)] == ["g#1"]       # r1 is not gold either: only the gold stays
    assert fit_budget([P("g#1"), P("g#2")], is_gold, size, 10) is None           # the gold alone does not fit: row dropped


def test_full_gold_replaces_a_chunk_of_a_gold_article_by_the_whole_article_and_drops_its_other_chunks():
    from train.sft.ctx_mix import full_gold

    ctx = [P("r1"), P("g#1"), P("r3"), P("g#1")]            # two chunks of the same gold article
    whole = {"g#1": Passage(pid="WHOLE", doc="d", text="g#1")}
    out = full_gold(ctx, {"g#1"}, KEY, whole.get, random.Random(0))
    assert [p.pid for p in out] == ["r1", "WHOLE", "r3"]


def test_full_gold_puts_every_missing_gold_into_non_gold_slots_and_never_overwrites_a_gold_passage_or_rank_one():
    from train.sft.ctx_mix import full_gold

    store = {k: Passage(pid=k, doc="d", text=k) for k in ("g#1", "g#2", "g#3")}
    for seed in range(20):
        ctx = [P("r1"), P("g#1"), P("r3"), P("r4"), P("r5")]
        out = full_gold(ctx, {"g#1", "g#2", "g#3"}, lambda p: p.text if p.text.startswith("g#") else p.pid, store.get, random.Random(seed))
        ids = [p.pid for p in out]
        assert ids[0] == "r1" and len(out) == 5
        assert {"g#1", "g#2", "g#3"} <= set(ids)


def test_full_gold_grows_the_context_when_there_are_more_gold_articles_than_slots():
    from train.sft.ctx_mix import full_gold

    store = {k: Passage(pid=k, doc="d", text=k) for k in ("g#1", "g#2", "g#3", "g#4")}
    ctx = [P("r1"), P("r2")]
    out = full_gold(ctx, set(store), lambda p: p.pid, store.get, random.Random(0))
    assert {p.pid for p in out} >= set(store) and out[0].pid == "r1"


def test_place_gold_randomly_moves_gold_to_random_positions_and_keeps_the_non_gold_order():
    from train.sft.ctx_mix import place_gold_randomly

    is_gold = lambda p: p.pid.startswith("g")   # noqa: E731
    ctx = [P("g1"), P("r2"), P("r3"), P("g4"), P("r5"), P("r6")]
    positions = set()
    for seed in range(60):
        out = place_gold_randomly(ctx, is_gold, random.Random(seed))
        assert sorted(p.pid for p in out) == sorted(p.pid for p in ctx)
        assert [p.pid for p in out if not is_gold(p)] == ["r2", "r3", "r5", "r6"]      # non-gold keep their rank order
        positions |= {i for i, p in enumerate(out) if p.pid == "g1"}
    assert len(positions) >= 5                                                          # the gold reaches (almost) every slot
