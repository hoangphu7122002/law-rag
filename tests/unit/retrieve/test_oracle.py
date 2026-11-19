from records import Passage
from retrieve.oracle import inject_gold


def P(i):
    return Passage(pid=str(i), doc="d", text=str(i))


def gold(p):
    return p.pid in {"g1", "g2"}


def test_gold_goes_to_rank_1_and_length_stays_k():
    out = inject_gold([P(1), P(2), P(3)], [P(1), P("g1")], gold)
    assert [p.pid for p in out] == ["g1", "1", "2"]


def test_list_with_gold_untouched():
    top = [P(1), P("g2"), P(3)]
    assert inject_gold(top, [P("g1")], gold) == top


def test_no_gold_in_pool_unchanged():
    top = [P(1), P(2)]
    assert inject_gold(top, [P(5), P(6)], gold) == top


def test_first_gold_in_pool_order():
    out = inject_gold([P(1), P(2)], [P(9), P("g2"), P("g1")], gold)
    assert out[0].pid == "g2"
