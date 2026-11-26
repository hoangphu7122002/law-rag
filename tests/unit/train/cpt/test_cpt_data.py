from train.cpt.cpt_data import docs_text, pack


def test_pack_concatenates_documents_with_eos_and_cuts_fixed_length_sequences():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    seqs = pack(docs, seq_len=4, eos=0)
    assert seqs == [[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]]


def test_pack_drops_only_the_incomplete_tail():
    seqs = pack([[1, 2, 3, 4, 5]], seq_len=4, eos=0)
    assert seqs == [[1, 2, 3, 4]]


def test_docs_text_orders_the_chunks_of_one_document_and_keeps_only_the_wanted_documents():
    nodes = [("a", 1, "Điều 2. B"), ("b", 1, "other"), ("a", 0, "Luật A\nĐiều 1. A")]
    out = docs_text(nodes, wanted={"a"})
    assert out == {"a": "Luật A\nĐiều 1. A\nĐiều 2. B"}
