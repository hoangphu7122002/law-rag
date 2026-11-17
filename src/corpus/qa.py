"""The questions of one split, as Question records."""
import json
from pathlib import Path

from records import Question


class QASet:
    def __init__(self, processed: Path = Path("data/processed")):
        self.processed = Path(processed)

    def load(self, split: str, limit: int | None = None) -> list[Question]:
        import polars as pl

        ids = json.loads((self.processed / "split.json").read_text())[split]
        qa = pl.read_parquet(self.processed / "qa.parquet").filter(pl.col("qid").is_in(ids)).sort("qid")
        rows = qa.head(limit) if limit else qa
        return [Question(r["qid"], r["question"], r["answer"], r["law_id"] or "") for r in rows.iter_rows(named=True)]
