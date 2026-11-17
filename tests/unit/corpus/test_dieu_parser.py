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
