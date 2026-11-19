from corpus.gold import GoldSet
from corpus.keys import law_key, stem_key
from corpus.sources.citations import Citations


def test_keys_agree_across_spellings():
    assert law_key("50/2024/TT-NHNN") == stem_key("50_2024_TT-NHNN") == stem_key("50_2024_TT-NHNN__285123")
    assert law_key("230/QĐ-UBND") == stem_key("230_QĐ-UBND_1")
    assert law_key("2124/QĐ-TTg") == law_key("2124/QĐ-TT")
    assert stem_key("72_2025_QH15") == "72/2025/QH15"


def test_spans_keep_order_and_year_hint():
    refs = Citations.spans("khoản 1 Điều 1 Quyết định 2124/QĐ-TTg năm 2011 và Thông tư 39/2016/TT-NHNN")
    assert [(r.ref, r.year) for r in refs] == [("2124/QĐ-TTg", 2011), ("39/2016/TT-NHNN", None)]


def test_each_article_goes_to_the_next_cited_document():
    g = GoldSet()
    g._cit = Citations.__new__(Citations)  # number refs resolve without the QA / vbpl files
    docs, pairs = g.parse("Căn cứ Điều 14 Thông tư 19/2016/TT-NHNN sửa đổi bởi khoản 5 Điều 1 Thông tư 26/2017/TT-NHNN. "
                          "Điều này áp dụng …", law_id="19/2016/TT-NHNN")
    assert docs == {"19/2016/TT-NHNN", "26/2017/TT-NHNN"}
    assert pairs == {"19/2016/TT-NHNN#14", "26/2017/TT-NHNN#1"}
