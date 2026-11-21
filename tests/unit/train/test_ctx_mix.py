import random

from records import Passage
from train.ctx_mix import drop_gold, inject_all, maybe_shuffle, missing_gold


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
