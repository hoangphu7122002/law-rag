from kg import KG, render_related


def graph():
    kg = KG()
    kg.upsert_doc("r", "75/2015/QH13", title="Luật Mặt trận")
    for i in range(7):
        kg.upsert_doc(f"l{i}", f"{i}/2020/QH", title="T" * 200, status="Còn hiệu lực")
        kg.add_edge(f"l{i}", "r", "amends")
    return kg


def test_unknown_law_says_so():
    assert "Không tìm thấy" in render_related("nope", KG().related("nope"))


def test_no_relation_says_so():
    kg = KG()
    kg.upsert_doc("r", "1/2020")
    assert "Không có" in render_related("1/2020", kg.related("1/2020"))


def test_each_document_is_one_short_line_with_family_and_status():
    text = render_related("75/2015/QH13", graph().related("75/2015/QH13", per_hop=2))
    lines = [l for l in text.splitlines() if l.startswith("- ")]
    assert len(lines) == 2
    assert all("amends" in l and "Còn hiệu lực" in l and len(l) < 160 for l in lines)   # title is cut


def test_truncation_is_stated_with_the_true_count_so_the_model_can_see_the_hop_is_large():
    text = render_related("75/2015/QH13", graph().related("75/2015/QH13", per_hop=2))
    assert "hop 1: 7 văn bản" in text and "hiện 2" in text
