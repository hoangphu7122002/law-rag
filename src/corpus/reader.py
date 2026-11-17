"""Law files -> LlamaIndex Documents. metadata["doc"] = file stem = law id with "/" -> "_"."""
from pathlib import Path

from llama_index.core import Document


class CorpusReader:
    def __init__(self, folder: Path = Path("data/processed/corpus"), lowercase: bool = False, flatten: bool = False):
        self.folder, self.lowercase, self.flatten = Path(folder), lowercase, flatten

    def files(self) -> list[Path]:
        return sorted(self.folder.glob("*.txt"))

    def read(self, limit: int | None = None) -> list[Document]:
        """limit = first N files only (smoke runs)."""
        return [self._document(f) for f in self.files()[:limit]]

    def _document(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if self.flatten:      # course baseline: newlines removed, lower-cased
            text = text.replace("\n", " ")
        if self.lowercase:
            text = text.lower()
        return Document(text=text.strip(), id_=path.stem, metadata={"doc": path.stem},
                        excluded_embed_metadata_keys=["doc"], excluded_llm_metadata_keys=["doc"])
