"""Documents cited in the QA answers (all splits) -> what the corpus must contain.

Answers cite documents by number ("Quyết định 2124/QĐ-TTg", "Thông tư 50/2024/TT-NHNN") or, for laws, by name and
year ("Luật Các tổ chức tín dụng 2024", "Bộ luật Dân sự 2015"). Each citation is resolved against the course
corpus, then vbpl metadata (by number; laws by title + year); the rest is left for the web crawler (luatvietnam.py).
Note: reads test answers (user decision 2026-11-18, docs/CHANGELOG.md); the result only selects corpus documents,
never training data.

    Citations().table()      -> polars DataFrame: ref, kind, number, name, year, n_questions, where, vbpl_id
    Citations.spans(answer)  -> [Ref(start, end, kind, ref, year)] in text order (used by corpus/gold.py)
    Citations().resolve(ref) -> law number or None (laws by name + year via vbpl titles)
"""
import re
import unicodedata
from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

from corpus.sources.vbpl import RAW, VbplImporter

QA = Path("data/processed/qa.parquet")
COURSE = Path("data/processed/corpus")
DOC_TYPES = r"(?:Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Nghị quyết|Chỉ thị|Công văn|Thông báo|Văn bản hợp nhất|Pháp lệnh|Lệnh|Hướng dẫn|Kế hoạch|Quy chế)"
YEAR_AFTER = r"(?:\s+(?:năm|ngày\s+[\d/]+\s*/)\s*((?:19|20)\d{2}))?"
NUMBER = re.compile(rf"{DOC_TYPES}\s+(?:số\s+)?(\d+[a-zA-Z]?/(?:\d{{4}}/)?[A-ZĐ][\wĐđ/-]*[A-ZĐa-zđ0-9]){YEAR_AFTER}")
BARE_NUMBER = re.compile(rf"\b(\d+[a-zA-Z]?/\d{{4}}/[A-ZĐ][\wĐđ-]*[A-ZĐa-zđ0-9]){YEAR_AFTER}")
LAW = re.compile(r"\b(Bộ luật|Luật)\s+([A-ZĐÂ][^\d,.;:()\[\]\n]{2,90}?)\s+(?:năm\s+)?((?:19|20)\d{2})\b")


def _norm(s: str) -> str:
    """Law name for matching: lower, no "Bộ luật"/"Luật" prefix, no "số <number>" suffix."""
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().lower()
    return re.sub(r"^(?:bộ luật|luật)\s+|\s+số\s+\S+$", "", s)


class Ref(NamedTuple):
    start: int
    end: int
    kind: str           # "number" | "law"
    ref: str            # law number, or "Luật <name>"
    year: int | None    # law year; for numbers only when the text says "… năm 2011"


class Citations:
    def __init__(self, qa: Path = QA):
        import polars as pl

        self.qa = pl.read_parquet(qa)

    @staticmethod
    def spans(answer: str) -> list[Ref]:
        """Every document citation in the (NFC) answer, in text order."""
        text = unicodedata.normalize("NFC", answer or "")
        out = {}
        for pat in (NUMBER, BARE_NUMBER):
            for m in pat.finditer(text):
                year = int(m.group(2)) if m.group(2) else None
                out.setdefault(m.start(1), Ref(m.start(1), m.end(), "number", m.group(1).rstrip("/-"), year))
        for m in LAW.finditer(text):
            name = re.sub(r"\s+(?:sửa đổi.*|hiện hành|mới nhất)$", "", m.group(2).strip())
            out[m.start()] = Ref(m.start(), m.end(), "law", f"{m.group(1)} {name}", int(m.group(3)))
        return [out[k] for k in sorted(out)]

    def refs(self, answer: str) -> set[tuple]:
        """(kind, number or name, year) for each document an answer cites; numbers without year."""
        return {(r.kind, r.ref, r.year if r.kind == "law" else None) for r in self.spans(answer)}

    @cached_property
    def _vbpl(self):
        import polars as pl

        return pl.read_parquet(RAW / "metadata.parquet").filter(pl.col("so_ky_hieu").is_not_null()
                                                               & (pl.col("loai_van_ban") != "Bản dịch văn bản"))

    @cached_property
    def _titles(self) -> dict:
        return self._law_titles(self._vbpl)

    def resolve(self, ref: Ref | tuple) -> str | None:
        """Law number of a citation: numbers as written, laws by vbpl title + year."""
        kind, name, year = (ref.kind, ref.ref, ref.year) if isinstance(ref, Ref) else ref
        if kind == "number":
            return name
        hit = self._titles.get((_norm(name), year))
        return hit[0] if hit else None

    def table(self):
        import polars as pl

        count = Counter(r for a in self.qa["answer"].to_list() for r in self.refs(a))
        key = VbplImporter.key
        course = {key(p.stem.replace("_", "/")) for p in COURSE.glob("*.txt")}
        meta = self._vbpl
        by_number = {key(n): i for n, i in zip(meta["so_ky_hieu"], meta["id"], strict=True)}
        by_title = self._titles
        rows = []
        for (kind, ref, year), n in count.items():
            where, vid, number = "web", None, ref if kind == "number" else None
            if kind == "law" and (hit := by_title.get((_norm(ref), year))):
                number, vid = hit
            if number and key(number) in course:
                where = "course"
            elif number and (vid := vid or by_number.get(key(number))):
                where = "vbpl"
            rows.append({"ref": ref if kind == "number" else f"{ref} {year}", "kind": kind, "number": number,
                         "year": year, "n_questions": n, "where": where, "vbpl_id": vid})
        return pl.DataFrame(rows).sort("n_questions", descending=True)

    @staticmethod
    def _law_titles(meta) -> dict[tuple[str, int], tuple[str, str]]:
        """("luật các tổ chức tín dụng", 2024) -> (number, vbpl id), from vbpl Luật / Bộ luật titles."""
        import polars as pl

        laws = meta.filter(pl.col("loai_van_ban").is_in(["Luật", "Bộ luật"]))
        out = {}
        for title, number, issued, vid in laws.select("title", "so_ky_hieu", "ngay_ban_hanh", "id").iter_rows():
            name = title or ""
            if issued and (y := re.search(r"(?:19|20)\d{2}$", issued)):
                out[(_norm(name), int(y.group(0)))] = (number, vid)
        return out
