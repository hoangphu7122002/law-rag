"""`make data`: unzip the corpus, clean the RES sheet, write the fixed train/dev/test split.

Inputs (course Drive folder "RAG Assignment"): data/raw/Dataset.zip, data/raw/res.csv
Outputs: data/processed/corpus/*.txt, data/processed/qa.parquet, data/processed/split.json
"""
import json
import random
import zipfile
from pathlib import Path
from typing import ClassVar

import polars as pl


class DatasetBuilder:
    SEED = 20261031
    N_TEST, N_DEV = 200, 300  # the assignment needs >= 100 test questions
    COLUMNS: ClassVar[dict[str, str]] = {
        "Câu Hỏi": "question",
        "Trả lời": "answer",
        "Số hiệu VBPL (Trích xuất)": "law_id",
        "Câu trả lời của URAx-Law": "urax_answer",
    }

    def __init__(self, raw: Path = Path("data/raw"), out: Path = Path("data/processed")):
        self.raw, self.out = Path(raw), Path(out)

    def build(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        n_files = self.unzip_corpus()
        qa = self.clean_qa()
        qa.write_parquet(self.out / "qa.parquet")
        split = self.split(qa.height)
        (self.out / "split.json").write_text(json.dumps(split))
        print(f"corpus files: {n_files}  qa pairs: {qa.height}  test/dev/train: "
              f"{len(split['test'])}/{len(split['dev'])}/{len(split['train'])}")

    def unzip_corpus(self) -> int:
        target = self.out / "corpus"
        if target.exists():
            return sum(1 for _ in target.glob("*.txt"))
        target.mkdir(parents=True)
        n = 0
        with zipfile.ZipFile(self.raw / "Dataset.zip") as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.endswith(".txt"):
                    continue
                (target / Path(info.filename).name).write_bytes(z.read(info))
                n += 1
        return n

    def clean_qa(self) -> pl.DataFrame:
        df = pl.read_csv(self.raw / "res.csv", infer_schema_length=0).select(list(self.COLUMNS)).rename(self.COLUMNS)
        df = df.filter(pl.col("question").is_not_null() & pl.col("answer").is_not_null())
        df = df.with_columns(pl.all().str.strip_chars()).unique(subset=["question"], keep="first", maintain_order=True)
        return df.with_row_index("qid")

    def split(self, n: int) -> dict[str, list[int]]:
        ids = list(range(n))
        random.Random(self.SEED).shuffle(ids)
        a, b = self.N_TEST, self.N_TEST + self.N_DEV
        return {"seed": self.SEED, "test": sorted(ids[:a]), "dev": sorted(ids[a:b]), "train": sorted(ids[b:])}
