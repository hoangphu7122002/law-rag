from generate.check import Gate
from records import Passage

P = lambda text: Passage("p", "50_2024_TT", text, 1.0)
CTX = [P("Điều 5. Phạm vi điều chỉnh\nNội dung điều 5."), P("Điều 12a. Trách nhiệm\nNội dung.")]


def test_a_gate_with_no_checks_passes_everything():
    assert Gate([]).passes("Căn cứ Điều 99 quy định", CTX) is True


def test_citation_must_appear_in_the_context():
    g = Gate(["grounded_citation"])
    assert g.passes("Căn cứ Điều 5 quy định về phạm vi.", CTX)
    assert g.passes("Theo Điều 12a, trách nhiệm là…", CTX)              # letter suffix kept
    assert not g.passes("Căn cứ Điều 99 quy định về phạm vi.", CTX)


def test_an_answer_that_cites_nothing_is_not_judged_by_this_check():
    assert Gate(["grounded_citation"]).passes("Không có điều khoản cụ thể.", CTX)


def test_one_ungrounded_citation_fails_even_if_another_is_grounded():
    assert not Gate(["grounded_citation"]).passes("Căn cứ Điều 5 và Điều 99.", CTX)


def test_unknown_check_name_is_an_error_not_a_silent_pass():
    import pytest

    with pytest.raises(ValueError, match="nope"):
        Gate(["nope"])
