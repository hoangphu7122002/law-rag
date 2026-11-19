from kg import KG
from kg.store import iso


def make():
    kg = KG()
    kg.upsert_doc("1", "10/2020/ND-CP", title="A", status="Hết hiệu lực toàn bộ", eff_from="2020-01-01", eff_to="2023-01-01")
    kg.upsert_doc("2", "20/2023/ND-CP", title="B", status="Còn hiệu lực", eff_from="2023-01-01")
    kg.upsert_doc("3", "30/2022/TT", title="C", eff_from="2022-01-01")
    kg.add_edge("2", "1", "replaces", "Thay thế")
    kg.add_edge("1", "2", "replaces", "Văn bản quy định hết hiệu lực")   # same pair seen from the other side: one undirected row per raw name
    kg.add_edge("3", "1", "guides", "Quy định chi tiết, hướng dẫn thi hành")
    kg.commit()
    return kg


def test_iso():
    assert iso("03/02/1997") == "1997-02-03" and iso(None) is None and iso("abc") is None


def test_edges_are_undirected_and_deduped():
    kg = make()
    assert kg.counts() == {"docs": 3, "edges": 3}
    kg.add_edge("2", "1", "replaces", "Thay thế")   # idempotent
    assert kg.counts()["edges"] == 3


def test_neighbors_filter_by_family():
    kg = make()
    assert {n["key"] for n in kg.neighbors("10/2020/ND-CP")} == {"20/2023/ND-CP", "30/2022/TT"}
    assert [n["key"] for n in kg.neighbors("10/2020/ND-CP", ("guides",))] == ["30/2022/TT"]
    assert {n["key"] for n in kg.amendments("10/2020/ND-CP")} == {"20/2023/ND-CP"}


def test_as_of_hides_documents_not_in_force_that_day():
    kg = make()
    assert kg.valid_on("1", "2021-06-01") is True and kg.valid_on("1", "2024-01-01") is False
    assert kg.valid_on("nope", "2021-01-01") is None
    assert "20/2023/ND-CP" not in {n["key"] for n in kg.neighbors("10/2020/ND-CP", as_of="2021-06-01")}


def test_update_and_delete():
    kg = make()
    kg.upsert_doc("3", "30/2022/TT", title="C v2", status="Còn hiệu lực", eff_from="2022-01-01")
    assert kg.doc("3")["title"] == "C v2" and kg.counts()["docs"] == 3
    kg.remove_doc("3")
    assert kg.doc("3") is None and kg.counts() == {"docs": 2, "edges": 2}


def test_same_law_number_different_documents():
    kg = make()
    kg.upsert_doc("9", "30/2022/TT", title="other province")
    assert sorted(kg.ids("30/2022/TT")) == ["3", "9"]


def test_documents_without_a_law_number_are_not_neighbours_of_each_other():
    kg = KG()
    kg.upsert_doc("a", "", title="no number 1")
    kg.upsert_doc("b", "", title="no number 2")
    kg.add_edge("a", "b", "cites")
    assert kg.neighbors("") == [] and kg.ids("") == []


def test_validity_window_starts_inclusive_and_ends_exclusive():
    kg = KG()
    kg.upsert_doc("1", "1/2020", eff_from="2020-01-01", eff_to="2023-01-01")
    assert kg.valid_on("1", "2019-12-31") is False
    assert kg.valid_on("1", "2020-01-01") is True
    assert kg.valid_on("1", "2022-12-31") is True
    assert kg.valid_on("1", "2023-01-01") is False


def test_open_ended_validity_windows():
    kg = KG()
    kg.upsert_doc("1", "1/2020", eff_from="2020-01-01")        # still in force
    kg.upsert_doc("2", "2/2020", eff_to="2020-01-01")          # start date unknown
    assert kg.valid_on("1", "2099-01-01") is True and kg.valid_on("1", "2019-01-01") is False
    assert kg.valid_on("2", "1990-01-01") is True and kg.valid_on("2", "2020-01-01") is False


def test_neighbours_with_unknown_dates_are_kept_by_as_of():
    kg = KG()
    kg.upsert_doc("1", "1/2020", eff_from="2020-01-01")
    kg.upsert_doc("2", "2/2020")                                # no dates at all
    kg.add_edge("1", "2", "cites")
    assert [n["key"] for n in kg.neighbors("1/2020", as_of="2021-01-01")] == ["2/2020"]


VBPL_RELATIONS = {  # all 31 names in data/raw/vbpl/relationships.parquet -> the family the KG must give them
    "Sửa đổi, bổ sung": "amends", "Văn bản bổ sung": "amends", "Văn bản sửa đổi": "amends", "Văn bản được bổ sung": "amends",
    "Văn bản được sửa đổi": "amends", "Đính chính": "amends",
    "Thay thế": "replaces", "Hợp nhất": "replaces", "Bãi bỏ": "repeals",
    "Văn bản quy định hết hiệu lực": "expires", "Văn bản hết hiệu lực": "expires", "Văn bản quy định hết hiệu lực 1 phần": "expires",
    "Văn bản bị hết hiệu lực 1 phần": "expires", "Tạm ngưng hiệu lực": "expires", "Đình chỉ thi hành": "expires",
    "Văn bản đình chỉ": "expires", "Văn bản đình chỉ 1 phần": "expires", "Văn bản bị đình chỉ": "expires", "Văn bản bị đình chỉ 1 phần": "expires",
    "Quy định chi tiết, hướng dẫn thi hành": "guides", "Văn bản HD, QĐ chi tiết": "guides", "Văn bản được HD, QĐ chi tiết": "guides",
    "Hướng dẫn áp dụng": "guides", "Giải thích": "guides",
    "Căn cứ": "bases_on", "Văn bản căn cứ": "bases_on", "Dẫn chiếu": "cites", "Văn bản dẫn chiếu": "cites",
    "Văn bản liên quan khác": "other", "Bản dịch": "other", "Công bố": "other",
}


def test_every_vbpl_relation_name_has_the_expected_family():
    from kg import FAMILY

    assert len(VBPL_RELATIONS) == 31
    assert {name: FAMILY.get(name, "other") for name in VBPL_RELATIONS} == VBPL_RELATIONS
    assert FAMILY.get("tên lạ", "other") == "other"


def vbpl_fixture(tmp_path):
    import polars as pl

    meta = pl.DataFrame({
        "id": ["1", "2", "3"], "so_ky_hieu": ["10/2020/NĐ-CP", "20/2023/NĐ-CP", None],
        "title": ["A", "B", "C"], "co_quan_ban_hanh": ["Chính phủ", "Chính phủ", "UBND"],
        "tinh_trang_hieu_luc": ["Hết hiệu lực toàn bộ", "Còn hiệu lực", "Chưa xác định"],
        "ngay_co_hieu_luc": ["01/01/2020", "01/01/2023", None], "ngay_het_hieu_luc": ["01/01/2023", None, None]})
    rel = pl.DataFrame({"doc_id": ["2", "1", "2", "2"], "other_doc_id": ["1", "2", "99", "3"],
                        "relationship": ["Thay thế", "Văn bản quy định hết hiệu lực", "Căn cứ", "Bản dịch"]})
    meta.write_parquet(tmp_path / "metadata.parquet")
    rel.write_parquet(tmp_path / "relationships.parquet")
    return tmp_path


def test_build_from_vbpl_tables(tmp_path):
    kg = KG.build(vbpl_fixture(tmp_path), tmp_path / "kg.sqlite")
    assert kg.counts() == {"docs": 3, "edges": 3}                      # the edge to unknown document 99 is dropped
    d = kg.doc("1")
    assert (d["key"], d["issuer"], d["status"], d["eff_from"], d["eff_to"]) == ("10/2020/NĐ-CP", "Chính phủ", "Hết hiệu lực toàn bộ", "2020-01-01", "2023-01-01")
    assert kg.doc("3")["key"] == "" and kg.doc("3")["eff_from"] is None
    assert {n["family"] for n in kg.neighbors("10/2020/NĐ-CP")} == {"replaces", "expires"}
    assert [n["key"] for n in kg.amendments("20/2023/NĐ-CP")] == ["10/2020/NĐ-CP", "10/2020/NĐ-CP"]


def test_graph_survives_reopening_the_file(tmp_path):
    KG.build(vbpl_fixture(tmp_path), tmp_path / "kg.sqlite")
    again = KG(tmp_path / "kg.sqlite")
    assert again.counts() == {"docs": 3, "edges": 3}
    again.upsert_doc("4", "40/2024/TT")
    again.commit()
    assert KG(tmp_path / "kg.sqlite").counts()["docs"] == 4


def test_a_neighbour_appears_once_per_family_even_when_vbpl_lists_the_relation_on_both_sides():
    kg = make()    # doc 1 / doc 2 are linked by "Thay thế" AND "Văn bản quy định hết hiệu lực" (both: replaces)
    rows = kg.neighbors("10/2020/ND-CP", ("replaces",))
    assert [(r["id"], r["family"]) for r in rows] == [("2", "replaces")]
    assert set(rows[0]["raw"].split(" | ")) == {"Thay thế", "Văn bản quy định hết hiệu lực"}


def test_removing_a_document_removes_its_edges_from_every_neighbour_view():
    kg = make()
    kg.remove_doc("1")
    assert kg.neighbors("20/2023/ND-CP") == [] and kg.neighbors("30/2022/TT") == []
    assert kg.counts() == {"docs": 2, "edges": 0}


def test_self_edges_are_ignored():
    kg = make()
    kg.add_edge("1", "1", "cites")
    assert kg.counts()["edges"] == 3


def test_neighbours_respect_the_limit():
    kg = KG()
    kg.upsert_doc("0", "0/2020")
    for i in range(1, 8):
        kg.upsert_doc(str(i), f"{i}/2020")
        kg.add_edge("0", str(i), "cites")
    assert len(kg.neighbors("0/2020", limit=3)) == 3
    assert len(kg.neighbors("0/2020")) == 7


def chain():
    """A <-amends- B <-amends- C, and A -guides-> G.  keys: a, b, c, g."""
    kg = KG()
    for i, k in (("1", "a"), ("2", "b"), ("3", "c"), ("4", "g")):
        kg.upsert_doc(i, k, title=f"doc {k}", status="Còn hiệu lực")
    kg.add_edge("2", "1", "amends", "Sửa đổi, bổ sung")
    kg.add_edge("3", "2", "amends", "Sửa đổi, bổ sung")
    kg.add_edge("1", "4", "guides", "Quy định chi tiết, hướng dẫn thi hành")
    return kg


def test_related_lists_documents_by_hop_and_never_revisits():
    out = chain().related("a", hops=2, families=("amends",))
    assert [(h["hop"], [r["key"] for r in h["docs"]]) for h in out] == [(1, ["b"]), (2, ["c"])]   # "a" itself and "b" are not listed again
    assert [h["total"] for h in out] == [1, 1]


def test_related_hops_defaults_to_one_and_family_filter_applies():
    kg = chain()
    assert [h["hop"] for h in kg.related("a", families=("amends",))] == [1]
    assert [r["key"] for r in kg.related("a", families=("guides",))[0]["docs"]] == ["g"]


def star(n_leaves=10):
    """root "r" amended by n leaves l0..; leaf l0 is itself amended by 4 documents m0.. (hop 2)."""
    kg = KG()
    kg.upsert_doc("r", "r")
    for i in range(n_leaves):
        kg.upsert_doc(f"l{i}", f"l{i}")
        kg.add_edge(f"l{i}", "r", "amends")
    for j in range(4):
        kg.upsert_doc(f"m{j}", f"m{j}")
        kg.add_edge(f"m{j}", "l0", "amends")
    return kg


def test_related_shows_at_most_per_hop_documents_but_reports_the_true_total():
    out = star().related("r", hops=2, per_hop=3, families=("amends",))
    assert [(h["hop"], h["total"], len(h["docs"])) for h in out] == [(1, 10, 3), (2, 4, 3)]


def test_related_never_shows_more_than_max_total_documents_across_hops():
    out = star().related("r", hops=2, per_hop=5, max_total=7, families=("amends",))
    assert sum(len(h["docs"]) for h in out) == 7
    assert [h["total"] for h in out] == [10, 4]               # totals stay truthful even when nothing more is shown
    assert len(out[1]["docs"]) == 2


def test_related_distinguishes_an_unknown_law_from_a_law_without_relations():
    kg = chain()
    kg.upsert_doc("9", "z", title="alone")
    assert kg.related("nope") is None
    assert kg.related("z") == []
