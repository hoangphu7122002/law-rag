from corpus.ingest.splitters import pack_sentences


def test_sentence_packing_never_cuts_inside_a_sentence_and_respects_the_limit():
    body = "Câu một dài. Câu hai cũng dài. Câu ba. Câu bốn rất rất dài."
    pieces = pack_sentences(body, max_chars=30)
    assert "".join(p.replace(" ", "") for p in pieces) == body.replace(" ", "")
    assert all(len(p) <= 30 for p in pieces)
    assert all(p.rstrip().endswith(".") for p in pieces)


def test_a_single_overlong_sentence_is_kept_whole():
    pieces = pack_sentences("Một câu rất rất rất dài không có dấu chấm giữa chừng.", max_chars=10)
    assert len(pieces) == 1


def test_list_markers_stay_with_their_text():
    from corpus.ingest.splitters import sentences

    assert sentences("1. Nội dung một. 2. Nội dung hai.") == ["1. Nội dung một.", "2. Nội dung hai."]


def test_khoan_units_keep_points_with_their_clause_and_merge_short_clauses():
    from corpus.ingest.splitters import pack_khoan

    body = "Điều 5. Tên\n1. Khoản một:\na) điểm a;\nb) điểm b.\n2. Khoản hai.\n3. Khoản ba."
    assert pack_khoan(body, max_chars=10_000) == [body]
    pieces = pack_khoan(body, max_chars=60)
    assert pieces[0].startswith("Điều 5. Tên\n1. Khoản một:\na) điểm a;\nb) điểm b.")
    assert any(p.startswith("2. Khoản hai.") or p.startswith("3.") for p in pieces[1:])
    assert all(len(p) <= 60 or p.startswith("Điều") for p in pieces)


def test_a_clause_longer_than_the_limit_falls_back_to_sentences():
    from corpus.ingest.splitters import pack_khoan

    body = "1. " + " ".join(f"Câu số {i} dài dài dài." for i in range(8))
    pieces = pack_khoan(body, max_chars=60)
    assert len(pieces) > 1 and all(len(p) <= 60 for p in pieces)


def _topic_embed(texts):
    return [[1.0, 0.0] if "thuế" in t else [0.0, 1.0] for t in texts]


def test_semantic_split_cuts_where_the_topic_changes():
    from corpus.ingest.splitters import semantic_split

    body = "\n".join(["Thuế một thuế.", "Thuế hai thuế.", "Thuế ba thuế.", "Xe một.", "Xe hai.", "Xe ba."]).replace("Thuế", "thuế")
    body = body.replace("thuế một thuế", "thuế một thuế")
    pieces = semantic_split(body, _topic_embed, pct=70, max_chars=10_000)
    assert len(pieces) == 2 and "thuế" in pieces[0] and "Xe" in pieces[1]


def test_semantic_split_respects_max_chars_and_handles_one_sentence():
    from corpus.ingest.splitters import semantic_split

    body = "\n".join(f"thuế câu {i} dài dài dài." for i in range(10))
    assert all(len(p) <= 60 for p in semantic_split(body, _topic_embed, pct=90, max_chars=60))
    assert semantic_split("Một câu.", _topic_embed, pct=90, max_chars=100) == ["Một câu."]
