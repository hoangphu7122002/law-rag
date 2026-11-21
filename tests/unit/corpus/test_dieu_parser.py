from llama_index.core import Document

from corpus import DieuNodeParser

LAW = """THÔNG TƯ

Quy định về
một việc

Điều 1. Phạm vi
điều chỉnh

1. Nội dung một.

Điều 2. Đối tượng

Cơ quan, tổ chức.
"""


def test_unwrap_joins_lines_inside_a_paragraph():
    assert DieuNodeParser.unwrap("Quy định về\nmột việc\n\nĐiều 1.") == "Quy định về một việc\nĐiều 1."


def test_split_articles_keeps_preamble_and_headings():
    parts = DieuNodeParser.split_articles(LAW)
    assert [h for h, _ in parts] == ["", "Điều 1. Phạm vi điều chỉnh", "Điều 2. Đối tượng"]
    assert parts[1][1].endswith("1. Nội dung một.")


def test_long_article_is_split_and_pieces_keep_the_heading():
    body = "Điều 9. Dài\n" + "\n\n".join(f"Khoản {i} " + "chữ " * 60 for i in range(10))
    nodes = DieuNodeParser(max_chars=400, overlap=0).get_nodes_from_documents(
        [Document(text=body, metadata={"doc": "x"})])
    assert len(nodes) > 1
    assert all(n.get_content().startswith("Điều 9. Dài") for n in nodes)
    assert all(n.metadata["doc"] == "x" for n in nodes)


def test_nodes_carry_article_heading_and_chapter():
    from llama_index.core import Document

    from corpus.ingest import DieuNodeParser

    text = "Chương I\nQUY ĐỊNH CHUNG\nĐiều 1. Phạm vi\nNội dung một.\nChương II. QUẢN LÝ\nĐiều 2. Trách nhiệm\nHai."
    nodes = DieuNodeParser().get_nodes_from_documents([Document(text=text, metadata={"doc": "x"})])
    by_article = {n.metadata["article"]: n.metadata for n in nodes}
    assert by_article["2"]["heading"] == "Điều 2. Trách nhiệm"
    assert by_article["2"]["chapter"] == "Chương II. QUẢN LÝ"
    assert by_article["1"]["chapter"] == "Chương I"
    assert "article" in nodes[0].excluded_embed_metadata_keys


def test_split_modes_cut_long_articles_at_sentence_or_clause_boundaries():
    body = "Điều 9. Dài\n" + "\n".join(f"{i}. " + "Câu dài dài dài. " * 6 for i in range(1, 7))
    doc = [Document(text=body, metadata={"doc": "x"})]
    for mode in ("sentence", "khoan"):
        nodes = DieuNodeParser(max_chars=300, overlap=0, split=mode).get_nodes_from_documents(doc)
        assert len(nodes) > 1, mode
        assert all(n.get_content().startswith("Điều 9. Dài") for n in nodes), mode
        assert all(n.get_content().rstrip().endswith(".") for n in nodes), mode
        assert all(n.metadata["article"] == "9" for n in nodes), mode


def test_semantic_mode_uses_the_given_embedder():
    body = "Điều 9. Dài\n" + "\n".join(["thuế một.", "thuế hai.", "xe một.", "xe hai."])
    parser = DieuNodeParser(max_chars=10, overlap=0, split="semantic", semantic_pct=60)
    parser.set_embedder(lambda ts: [[1.0, 0.0] if "thuế" in t else [0.0, 1.0] for t in ts])
    nodes = parser.get_nodes_from_documents([Document(text=body, metadata={"doc": "x"})])
    texts = [n.get_content() for n in nodes]
    assert any("thuế" in t and "xe" not in t for t in texts) and any("xe" in t and "thuế" not in t for t in texts)
