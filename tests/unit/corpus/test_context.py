from corpus.context import chapter_of

SEPARATE = """Luật Các tổ chức tín dụng số 32/2024/QH15

Chương I
NHỮNG QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
Luật này quy định về tổ chức tín dụng.
Điều 2. Giải thích từ ngữ
Trong Luật này, theo Chương IV của Luật khác thì áp dụng.
Chương IV TỔ CHỨC, QUẢN TRỊ, ĐIỀU HÀNH CỦA TỔ CHỨC TÍN DỤNG
Điều 40. Hội đồng quản trị
Nội dung.
"""


def test_chapter_title_on_next_line():
    assert chapter_of(SEPARATE, "2") == "Chương I: Những quy định chung"


def test_chapter_title_on_same_line_and_mention_in_sentence_ignored():
    assert chapter_of(SEPARATE, "40") == "Chương IV: Tổ chức, quản trị, điều hành của tổ chức tín dụng"


def test_muc_under_chuong_and_reset_on_new_chuong():
    doc = "Nghị định số 1/2020/NĐ-CP\nChương II\nQUY ĐỊNH CỤ THỂ\nMục 2. Cấp phép\nĐiều 5. A\nx\nChương III Xử lý\nĐiều 6. B\ny\n"
    assert chapter_of(doc, "5") == "Chương II: Quy định cụ thể / Mục 2: Cấp phép"
    assert chapter_of(doc, "6") == "Chương III: Xử lý"


def test_no_chapters_or_missing_dieu_gives_empty():
    doc = "Thông tư số 1/2020/TT-BTC\nĐiều 1. Phạm vi\nx\n"
    assert chapter_of(doc, "1") == ""
    assert chapter_of(SEPARATE, "99") == ""


def test_duplicate_dieu_number_takes_first_occurrence():
    doc = "Chương I Chung\nĐiều 1. A\nx\nChương II Riêng\nĐiều 1. B\ny\n"
    assert chapter_of(doc, "1") == "Chương I: Chung"


from corpus.context import scope_of


def test_scope_first_sentence_of_dieu_1():
    assert scope_of(SEPARATE) == "Luật này quy định về tổ chức tín dụng."


def test_scope_absent_when_no_dieu_1():
    assert scope_of("Luật X\nĐiều 2. A\nbody\n") == ""


def test_scope_trimmed_at_word_boundary():
    doc = "Điều 1. Phạm vi\n" + "từ " * 100 + "hết.\n"
    s = scope_of(doc, max_chars=20)
    assert len(s) <= 20 and set(s.split()) == {"từ"}  # no half word, no trailing space
    assert s == s.strip()


from corpus.context import context_line, enrich


def test_context_line_drops_empty_parts_and_caps_length():
    assert context_line("Luật A", "Chương I: Chung", "Luật này quy định X.") == "Văn bản: Luật A | Phần: Chương I: Chung | Phạm vi: Luật này quy định X."
    assert context_line("Luật A", "", "") == "Văn bản: Luật A"
    assert len(context_line("T" * 300, "C" * 300, "S" * 300)) <= 400


def test_enrich_prepends_once():
    text = "Luật A\nĐiều 2. Giải thích\nbody"
    out = enrich(text, SEPARATE, "2", "Luật A")
    assert out.splitlines()[0].startswith("Văn bản: Luật A | Phần: Chương I: Những quy định chung")
    assert out.endswith(text)
    assert enrich(out, SEPARATE, "2", "Luật A") == out


from records import Passage
from corpus.context import pool_with_context


def test_pool_with_context_changes_only_text():
    pool = [Passage("n1", "32_2024_QH15", "Luật A\nĐiều 40. Hội đồng quản trị\nNội dung", 0.5),
            Passage("n2", "other", "Luật B\nĐiều 3. X\nbody", 0.1),
            Passage("n3", "32_2024_QH15", "no heading here", 0.0)]
    out = pool_with_context(pool, lambda d: SEPARATE if d == "32_2024_QH15" else None)
    assert [(p.pid, p.doc, p.score) for p in out] == [(p.pid, p.doc, p.score) for p in pool]
    assert out[0].text.startswith("Văn bản: Luật A | Phần: Chương IV: Tổ chức")
    assert out[0].text.endswith(pool[0].text)
    assert out[1].text == pool[1].text and out[2].text == pool[2].text
    assert pool[0].text.startswith("Luật A")   # input untouched
