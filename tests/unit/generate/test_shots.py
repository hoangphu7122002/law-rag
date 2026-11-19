from generate.prompt import Prompt
from generate.shots import ShotStore, words
from records import Passage


def store(*items) -> ShotStore:
    s = ShotStore.__new__(ShotStore)
    s.pool = [(t, a, words(t), frozenset(p)) for t, a, p in items]
    return s


POOL = [
    ("thời hạn hợp đồng lao động là bao lâu", "Tối đa 36 tháng", {("LD", "Điều 20")}),
    ("mức lương tối thiểu vùng là bao nhiêu", "Theo vùng", {("LD", "Điều 91")}),
    ("hợp đồng lao động có thời hạn", "Có thời hạn", {("LD", "Điều 20")}),
    ("thuế thu nhập cá nhân", "Thuế TNCN", {("TNCN", "Điều 1")}),
]


def test_messages_without_shots_unchanged():
    p = Prompt.load("vi_rag")
    passages = [Passage(text="ngữ cảnh A", doc="d")] if False else []
    base = p.messages("Câu hỏi gốc?", passages)
    assert base == p.messages("Câu hỏi gốc?", passages, None) == p.messages("Câu hỏi gốc?", passages, [])
    assert "Ví dụ" not in base[-1]["content"]


def test_messages_with_shots_prepends_examples_and_keeps_question_last():
    p = Prompt.load("vi_rag")
    shots = [("Q một?", "A một."), ("Q hai?", "A hai.")]
    user = p.messages("Câu hỏi gốc?", [], shots)[-1]["content"]
    assert user.startswith("Ví dụ câu hỏi")
    for q, a in shots:
        assert q in user and a in user
    assert user.index("Q hai?") < user.index("Câu hỏi gốc?")  # original task follows the shots


def test_pick_orders_by_jaccard_and_limits_n():
    s = store(*POOL)
    got = s.pick("hợp đồng lao động thời hạn", 2)
    assert got[0][0] == "hợp đồng lao động có thời hạn"
    assert len(got) == 2
    assert got[1][0] == "thời hạn hợp đồng lao động là bao lâu"
    assert got[0][1] == "Có thời hạn"


def test_pick_zero_returns_empty():
    assert store(*POOL).pick("hợp đồng lao động", 0) == []
    assert store(*POOL).pick("hợp đồng lao động", -1) == []


def test_pick_excludes_identical_question():
    s = store(*POOL)
    got = s.pick("thuế thu nhập cá nhân", 4)
    assert all(q != "thuế thu nhập cá nhân" for q, _ in got)
    assert len(got) == 3


def test_pick_avoid_excludes_pool_items_sharing_a_pair():
    s = store(*POOL)
    got = [q for q, _ in s.pick("hợp đồng lao động thời hạn", 4, avoid=frozenset({("LD", "Điều 20")}))]
    assert "hợp đồng lao động có thời hạn" not in got
    assert "thời hạn hợp đồng lao động là bao lâu" not in got
    assert sorted(got) == sorted(["mức lương tối thiểu vùng là bao nhiêu", "thuế thu nhập cá nhân"])


def test_pick_avoid_empty_keeps_all():
    s = store(*POOL)
    assert len(s.pick("hợp đồng lao động", 4, avoid=frozenset())) == 4
