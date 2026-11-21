from corpus.sources.luatvietnam import LuatVietnam, merge_rows


def test_merge_rows_keeps_old_rows_and_dedupes_by_file():
    old = [{"file": "a.txt", "ref": "A"}, {"file": "b.txt", "ref": "B"}]
    new = [{"file": "b.txt", "ref": "B2"}, {"file": "c.txt", "ref": "C"}]
    out = merge_rows(old, new)
    assert [r["file"] for r in out] == ["a.txt", "b.txt", "c.txt"]
    assert next(r for r in out if r["file"] == "b.txt")["ref"] == "B2"      # newer wins


def test_number_match_does_not_confuse_116_with_16():
    urls = ["https://luatvietnam.vn/x/thong-tu-116-2018-tt-btc-che-do-169896-d1.html",
            "https://luatvietnam.vn/x/thong-tu-16-2018-tt-btc-hang-hoa-111111-d1.html"]
    hits = LuatVietnam.__new__(LuatVietnam).match({"kind": "number", "number": "16/2018/TT-BTC", "ref": "16/2018/TT-BTC"}, urls)
    assert hits == [urls[1]]
