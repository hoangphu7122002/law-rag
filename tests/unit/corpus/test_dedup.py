from llama_index.core import Document

from corpus.ingest import DocDedup

BODY = " ".join(f"từ{i}" for i in range(200))


def doc(stem: str, text: str) -> Document:
    return Document(text=text, metadata={"doc": stem})


def test_drops_exact_and_same_number_copies_keeps_other_years():
    docs = [doc("02_NQ-CP", BODY), doc("x", BODY.upper()),                      # exact (case only)
            doc("02_NQ-CP__1", BODY + " thêm một dòng"),                      # same number, near copy
            doc("02_NQ-CP__2", " ".join(f"khác{i}" for i in range(200)))]     # same number, other text
    kept = DocDedup()(docs)
    assert [d.metadata["doc"] for d in kept] == ["02_NQ-CP", "02_NQ-CP__2"]
