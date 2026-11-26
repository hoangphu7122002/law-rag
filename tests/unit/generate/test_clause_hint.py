from generate.clause_hint import clause_hint, split_units
from records import Passage

ART17 = ("Thông tư X quy định về bảo hiểm xe\nĐiều 17. Xử phạt người điều khiển xe máy\n"
         "1. Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người không có bảo hiểm bắt buộc.\n"
         "2. Phạt tiền từ 300.000 đồng đối với người không mang theo giấy đăng ký xe:\n"
         "a) xe máy;\nb) xe mô tô.\n"
         "3. Tái phạm thì phạt gấp đôi.")


def test_split_units_cuts_an_article_into_numbered_khoan_with_their_items():
    units = split_units(Passage("p1", "50_2024_TT-NHNN", ART17))
    assert [(u.dieu, u.khoan) for u in units] == [("17", "1"), ("17", "2"), ("17", "3")]
    assert "b) xe mô tô." in units[1].text and units[0].doc == "50_2024_TT-NHNN"


def test_an_article_without_numbered_khoan_is_one_unit_without_khoan():
    units = split_units(Passage("p", "d", "Điều 5. Tên\nNội dung một đoạn duy nhất."))
    assert [(u.dieu, u.khoan) for u in units] == [("5", None)]


def test_hint_names_the_khoan_whose_words_overlap_the_question_most():
    h = clause_hint("Phạt bao nhiêu tiền nếu không có bảo hiểm bắt buộc?", [Passage("p1", "d", ART17)], top=2)
    assert h.splitlines()[0].startswith("Khoản có thể liên quan nhất:")
    assert "khoản 1 Điều 17" in h.split(";")[0]


def test_hint_is_empty_when_nothing_overlaps():
    assert clause_hint("zzz qqq", [Passage("p1", "d", ART17)], top=2) == ""
