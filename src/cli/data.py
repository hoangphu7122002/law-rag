"""make data: unzip the corpus, clean the RES sheet, write the fixed train/dev/test split.

Inputs (from the course Drive folder): data/raw/Dataset.zip, data/raw/res.csv
Outputs: data/processed/corpus/*.txt, data/processed/qa.parquet, data/processed/split.json
"""
import json
import random
import zipfile
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
OUT = Path("data/processed")
SEED = 20261031
N_TEST, N_DEV = 200, 100  # the assignment needs >= 100 test questions

COLUMNS = {
    "Câu Hỏi": "question",
    "Trả lời": "answer",
    "Số hiệu VBPL (Trích xuất)": "law_id",
    "Câu trả lời của URAx-Law": "urax_answer",
}


def unzip_corpus() -> int:
    target = OUT / "corpus"
    if target.exists():
        return sum(1 for _ in target.glob("*.txt"))
    target.mkdir(parents=True)
    n = 0
    with zipfile.ZipFile(RAW / "Dataset.zip") as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(".txt"):
                continue
            (target / Path(info.filename).name).write_bytes(z.read(info))
            n += 1
    return n


def clean_qa() -> pl.DataFrame:
    df = pl.read_csv(RAW / "res.csv", infer_schema_length=0).select(list(COLUMNS)).rename(COLUMNS)
    df = df.filter(pl.col("question").is_not_null() & pl.col("answer").is_not_null())
    df = df.with_columns(pl.all().str.strip_chars()).unique(subset=["question"], keep="first", maintain_order=True)
    return df.with_row_index("qid")


def split(n: int) -> dict[str, list[int]]:
    ids = list(range(n))
    random.Random(SEED).shuffle(ids)
    return {
        "seed": SEED,
        "test": sorted(ids[:N_TEST]),
        "dev": sorted(ids[N_TEST : N_TEST + N_DEV]),
        "train": sorted(ids[N_TEST + N_DEV :]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_files = unzip_corpus()
    qa = clean_qa()
    qa.write_parquet(OUT / "qa.parquet")
    s = split(qa.height)
    (OUT / "split.json").write_text(json.dumps(s))
    print(f"corpus files: {n_files}  qa pairs: {qa.height}  test/dev/train: "
          f"{len(s['test'])}/{len(s['dev'])}/{len(s['train'])}")


if __name__ == "__main__":
    main()
