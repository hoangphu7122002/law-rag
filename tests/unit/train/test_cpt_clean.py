from train.cpt_clean import Dedup, chunk_doc, mask_pii, normalize, quality, strip_html


def test_strip_html_keeps_block_structure_and_drops_scripts_and_entities():
    html = "<html><script>var x=1;</script><body><p>Điều 1. Phạm vi</p><p>Luật này &amp; nghị định</p><table><tr><td>a</td><td>b</td></tr></table></body></html>"
    out = strip_html(html)
    assert "var x" not in out and "<" not in out
    assert out.splitlines()[0] == "Điều 1. Phạm vi" and "Luật này & nghị định" in out


def test_normalize_fixes_spaces_control_chars_and_collapses_blank_lines():
    assert normalize("a b\x00c\r\n\n\n\n d  \n") == "a bc\n\nd"


def test_quality_rejects_short_non_vietnamese_mojibake_and_repetitive_text_but_keeps_a_law():
    law = ("Điều 5. Quyền và nghĩa vụ của người lao động\n" + "1. Người lao động có quyền được trả lương đầy đủ, đúng hạn theo thỏa thuận trong hợp đồng lao động. " * 6)
    assert quality(law) is None
    assert quality("Điều 1. Ngắn") == "too_short"
    assert quality("This is an English text about contracts and employment law. " * 20) == "not_vietnamese"
    assert quality(("Ã¡ng Ä‘Ã¢y lÃ  vÄƒn báº£n lá»—i mÃ£ hÃ³a " * 30)) == "mojibake"
    assert quality("Người lao động nghỉ việc.\n" * 60) == "repetitive"


def test_mask_pii_hides_phone_email_id_numbers_and_names_after_party_titles():
    t = "Bị cáo Nguyễn Văn An, sinh năm 1980, CCCD 001080012345, SĐT 0912345678, email an@gmail.com, bà Trần Thị Bình trình bày."
    out = mask_pii(t)
    assert "Nguyễn Văn An" not in out and "Trần Thị Bình" not in out
    assert "001080012345" not in out and "0912345678" not in out and "an@gmail.com" not in out
    assert "Bị cáo [TÊN]" in out and "sinh năm 1980" in out


def test_dedup_drops_repeated_documents_repeated_long_paragraphs_and_repeated_doc_numbers():
    d = Dedup(min_par=60, min_new=20)
    para = "Điều 3. Giải thích từ ngữ. Trong luật này các từ ngữ dưới đây được hiểu như sau, người lao động là người làm việc theo hợp đồng."
    assert d.accept("doc1", f"{para}\nPhần riêng của văn bản một, khác biệt.") is not None
    assert d.accept("doc2", f"{para}\nPhần riêng của văn bản hai, khác biệt rõ rệt hoàn toàn.") == "Phần riêng của văn bản hai, khác biệt rõ rệt hoàn toàn."
    assert d.accept("doc3", f"{para}\nPhần riêng của văn bản một, khác biệt.") is None   # nothing new left
    assert d.accept("doc4", "Một văn bản hoàn toàn mới và dài đủ để được giữ lại sau khi lọc trùng nội dung.", key="01/2020/NĐ-CP") is not None
    assert d.accept("doc5", "Bản sao khác nội dung nhưng cùng số hiệu văn bản đã gặp trước đó rồi.", key="01/2020/NĐ-CP") is None


def test_chunk_doc_cuts_on_paragraphs_prefixes_the_title_and_never_exceeds_the_limit():
    doc = "\n".join(f"Điều {i}. " + "nội dung " * 40 for i in range(1, 30))
    chunks = chunk_doc(doc, title="Luật Mẫu số 1/2020/QH14", max_chars=1500)
    assert len(chunks) > 3 and all(c.startswith("Luật Mẫu số 1/2020/QH14\n") for c in chunks)
    assert all(len(c) <= 1500 + 80 for c in chunks)
    assert "".join(c.split("\n", 1)[1] for c in chunks).replace("\n", "") == doc.replace("\n", "")


def test_mask_pii_also_hides_family_context_names_and_names_already_cut_to_an_initial():
    out = mask_pii("con ông Lê Văn Cường và bà Phạm Thị Dung; có chồng Nguyễn Hoàng D và 02 con; Quách Thị V sinh năm 1987")
    assert "Lê Văn Cường" not in out and "Phạm Thị Dung" not in out
    assert "Nguyễn Hoàng D" not in out and "Quách Thị V" not in out and "sinh năm 1987" in out
    assert "Hội đồng xét xử" in mask_pii("Hội đồng xét xử nhận định Công ty TNHH A")
