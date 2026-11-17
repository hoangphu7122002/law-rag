"""Extra corpus from vbpl.vn (national legal document portal, Ministry of Justice), via the HF dataset
th1nhng0/vietnamese-legal-documents (CC BY 4.0): metadata.parquet + content.parquet (HTML) in data/raw/vbpl/.

Selection is test-blind: a fixed rule on document metadata only (scope, type, year); no QA split is read.
Documents already in the course corpus (same number) are skipped, so the course text wins.

    VbplImporter().build()   ->  data/processed/vbpl/<number with / -> _>.txt + manifest.parquet
"""
import html
import re
import unicodedata
from pathlib import Path
from typing import ClassVar

RAW = Path("data/raw/vbpl")
OUT = Path("data/processed/vbpl")
COURSE = Path("data/processed/corpus")
HF = "https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents/resolve/main/data"


class VbplImporter:
    SCOPES: ClassVar[list[str]] = ["Trung ương", "Toàn quốc"]
    SKIP_TYPES: ClassVar[list[str]] = ["Bản dịch văn bản"]  # English translations

    def __init__(self, min_year: int = 2010):
        self.min_year = min_year

    def build(self) -> int:
        import polars as pl

        self.download()
        meta = self.select(pl.read_parquet(RAW / "metadata.parquet"))
        course = {self.key(p.stem.replace("_", "/")) for p in COURSE.glob("*.txt")}
        meta = meta.filter(~pl.col("so_ky_hieu").map_elements(self.key, return_dtype=pl.String).is_in(list(course)))
        docs = (pl.scan_parquet(RAW / "content.parquet").filter(pl.col("id").is_in(meta["id"].to_list()))
                .collect().join(meta, on="id"))
        OUT.mkdir(parents=True, exist_ok=True)
        rows, seen = [], set()
        for row in docs.iter_rows(named=True):
            page = row.pop("content_html")
            name = self.filename(row["so_ky_hieu"])
            text = self.to_text(page)
            if not name or name in seen or len(text) < 200:
                continue
            seen.add(name)
            (OUT / f"{name}.txt").write_text(f"{row['title']}\n\n{text}", encoding="utf-8")
            rows.append({**row, "file": f"{name}.txt"})
        pl.DataFrame(rows).write_parquet(OUT / "manifest.parquet")
        print(f"vbpl: {meta.height} selected, {len(rows)} written to {OUT} (course duplicates skipped: {len(course)} known)")
        return len(rows)

    def select(self, meta):
        import polars as pl

        year = pl.col("ngay_ban_hanh").str.slice(-4).cast(pl.Int32, strict=False)
        return meta.filter(pl.col("pham_vi").is_in(self.SCOPES) & ~pl.col("loai_van_ban").is_in(self.SKIP_TYPES)
                           & (year >= self.min_year) & pl.col("so_ky_hieu").is_not_null())

    @staticmethod
    def download() -> None:
        import urllib.request

        RAW.mkdir(parents=True, exist_ok=True)
        for name in ("metadata", "content", "relationships"):
            if not (RAW / f"{name}.parquet").exists():
                print(f"downloading {name}.parquet ...")
                urllib.request.urlretrieve(f"{HF}/{name}.parquet", RAW / f"{name}.parquet")

    @staticmethod
    def key(number: str) -> str:
        """Comparable law number: NFC, no spaces, upper, '/QĐ-TT' == '/QĐ-TTG'."""
        k = re.sub(r"\s+", "", unicodedata.normalize("NFC", number or "")).upper()
        return re.sub(r"-TTG$", "-TT", k)

    @staticmethod
    def filename(number: str) -> str:
        return re.sub(r"[^\wĐđ.-]+", "_", unicodedata.normalize("NFC", number.strip()).replace("/", "_")).strip("_")

    @staticmethod
    def to_text(page: str) -> str:
        body = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", "", page or "")
        body = re.sub(r"(?i)<br\s*/?>|</p>|</tr>|</h\d>|</li>|</div>", "\n", body)
        text = html.unescape(re.sub(r"<[^>]+>", "", body)).replace("\xa0", " ")
        lines = (re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines())
        return "\n".join(ln for ln in lines if ln)
