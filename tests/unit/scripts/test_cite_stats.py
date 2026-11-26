
from cite_stats import cite_stats

REF = "Căn cứ tại khoản 2 Điều 5 Thông tư 07 quy định mức phạt tiền như sau: mức phạt từ 1 đến 2 triệu đồng."


def test_matched_vs_mismatched_citation_buckets():
    rows = [
        {"ref": REF, "pred": "Căn cứ tại khoản 2 Điều 5 Thông tư 07 quy định mức phạt tiền như sau: mức phạt từ 1 đến 2 triệu đồng."},
        {"ref": REF, "pred": "Căn cứ Điều 9 quy định về việc khác hoàn toàn với câu hỏi đặt ra ở đây."},
    ]
    s = cite_stats(rows)
    assert s["n"] == 2 and s["match_share"] == 0.5
    assert s["rouge_match"] > s["rouge_mismatch"]
