"""Law files -> LlamaIndex Documents. metadata["doc"] = file stem = law id with "/" -> "_", ["source"] = folder.
Sources = folders under data/processed/: corpus (course, 16k files) and optionally vbpl (`make vbpl`).
vbpl documents also get their portal metadata (manifest.parquet) under the same keys LawCleaner uses for the
course info block: number, type, field, issuer, issued, effective, expired, status. All metadata is hidden from
embedding and LLM text."""
from pathlib import Path

from llama_index.core import Document

MANIFEST_KEYS = {"so_ky_hieu": "number", "loai_van_ban": "type", "linh_vuc": "field", "co_quan_ban_hanh": "issuer",
                 "ngay_ban_hanh": "issued", "ngay_co_hieu_luc": "effective", "ngay_het_hieu_luc": "expired",
                 "tinh_trang_hieu_luc": "status"}


class CorpusReader:
    def __init__(self, sources: list[str] = ("corpus",), lowercase: bool = False, flatten: bool = False):
        self.folders = [Path("data/processed") / s for s in sources]
        self.lowercase, self.flatten = lowercase, flatten
        self._manifests: dict[Path, dict[str, dict]] = {}

    def files(self) -> list[Path]:
        return [f for folder in self.folders for f in sorted(folder.glob("*.txt"))]

    def read(self, limit: int | None = None) -> list[Document]:
        """limit = first N files only (smoke runs)."""
        return [self._document(f) for f in self.files()[:limit]]

    def _document(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if self.flatten:      # course baseline: newlines removed, lower-cased
            text = text.replace("\n", " ")
        if self.lowercase:
            text = text.lower()
        meta = {"doc": path.stem, "source": path.parent.name, **self._manifest(path.parent).get(path.name, {})}
        return Document(text=text.strip(), id_=path.stem, metadata=meta,
                        excluded_embed_metadata_keys=list(meta), excluded_llm_metadata_keys=list(meta))

    def _manifest(self, folder: Path) -> dict[str, dict]:
        """file name -> metadata, from <folder>/manifest.parquet when it exists (vbpl)."""
        if folder not in self._manifests:
            path, rows = folder / "manifest.parquet", {}
            if path.exists():
                import polars as pl

                for r in pl.read_parquet(path).iter_rows(named=True):
                    rows[r["file"]] = {new: r[old] for old, new in MANIFEST_KEYS.items() if r.get(old)}
            self._manifests[folder] = rows
        return self._manifests[folder]
