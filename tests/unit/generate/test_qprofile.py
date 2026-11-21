from generate.qprofile import needs_fallback, question_profile


def test_full_law_number_is_extracted_as_a_key():
    p = question_profile("Ở đâu có thể tải mẫu quyết định chấp thuận việc sáp nhập quỹ tín dụng nhân dân theo Phụ lục kèm Thông tư 10/2025/TT-NHNN?")
    assert p["laws"] == ["10/2025/TT-NHNN"]


def test_short_number_without_the_issuer_is_a_partial_reference():
    p = question_profile("Dự thảo sửa đổi Thông tư 39/2016 vừa được ban hành gồm những điểm sửa đổi nào?")
    assert p["laws"] == [] and p["partial"] == ["39/2016"]


def test_article_and_named_law_with_year():
    p = question_profile("Luật quy định những nhiệm vụ gì cho Bộ Tài chính (tham chiếu Điều 51 Luật Phòng, chống rửa tiền 2022)?")
    assert p["dieu"] == ["51"]
    assert p["law_names"] == [("Luật Phòng, chống rửa tiền", 2022)]


def test_article_clause_ranges_and_lists():
    p = question_profile("Nội dung khoản 2 Điều 27 và Điều 28, 29 Thông tư 20/2024/TT-NHNN là gì?")
    assert p["dieu"] == ["27", "28", "29"] and p["laws"] == ["20/2024/TT-NHNN"]


def test_flags_form_appendix_draft_notice():
    p = question_profile("Dự thảo thông báo kèm phụ lục mẫu biểu có nội dung gì?")
    assert set(p["flags"]) >= {"du_thao", "thong_bao", "phu_luc", "mau_bieu"}


def test_question_type_vocabulary():
    assert question_profile("Trong những trường hợp nào tổ chức tín dụng bị thu hồi giấy phép?")["type"] == "truong_hop"
    assert question_profile("Thời hạn nộp hồ sơ là bao lâu?")["type"] == "thoi_han"
    assert question_profile("Cơ quan nào có thẩm quyền cấp giấy phép?")["type"] == "tham_quyen"
    assert question_profile("Mức phạt đối với hành vi này là bao nhiêu?")["type"] == "muc_phat"
    assert question_profile("Điều kiện để thành lập quỹ tín dụng nhân dân là gì?")["type"] == "dieu_kien"


def test_unspecific_question_needs_the_llm_fallback():
    p = question_profile("Ngân hàng được làm gì khi khách hàng không trả nợ?")
    assert p["laws"] == [] and p["dieu"] == [] and needs_fallback(p) is True


def test_specific_question_does_not_need_the_fallback():
    assert needs_fallback(question_profile("Điều 5 Thông tư 23/2014/TT-NHNN quy định gì?")) is False


def test_years_are_collected_without_the_law_number_year():
    p = question_profile("Lịch nghỉ Tết Âm lịch năm 2025 của ngân hàng như thế nào?")
    assert p["years"] == [2025]
