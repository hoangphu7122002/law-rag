"""Run = one directory under data/runs/; Ledger = data/runs/results.tsv, one row per scored run.

    data/runs/<exp>__<hash>/
      config.yaml        resolved config
      meta.json          git commit, backend, start time
      retrieved.jsonl    passages per question (copied from the retrieval cache)
      predictions.jsonl  one Prediction per line, appended as answers arrive (rerun = resume)
      trajectories.jsonl agent runs only: {qid, question, messages} per answered question
      metrics.json       summary + per-question scores, written by Evaluator
"""
import json
import subprocess
import time
from pathlib import Path
from typing import ClassVar

import yaml

from config import Config

RUNS = Path("data/runs")


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def create(cls, cfg: Config) -> "Run":
        run = cls(RUNS / f"{cfg.name}__{cfg.run_key()}")
        run.path.mkdir(parents=True, exist_ok=True)
        (run.path / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), allow_unicode=True, sort_keys=False))
        meta = {"git": git_commit(), "backend": cfg.backend.name, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (run.path / "meta.json").write_text(json.dumps(meta, indent=1))
        return run

    def config(self) -> dict:
        return yaml.safe_load((self.path / "config.yaml").read_text(encoding="utf-8"))

    def done_qids(self) -> set[int]:
        return {row["qid"] for row in self.predictions()}

    def predictions(self) -> list[dict]:
        rows = []
        p = self.path / "predictions.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # a line cut by a crash
                    pass
        return rows

    def append_prediction(self, row: dict) -> None:
        with open(self.path / "predictions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_trajectories(self, rows: list[dict]) -> None:
        with open(self.path / "trajectories.jsonl", "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_json(self, name: str, obj) -> None:
        (self.path / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))

    def read_json(self, name: str):
        return json.loads((self.path / name).read_text(encoding="utf-8"))


class Ledger:
    """data/runs/results.tsv: one row per scored run, fixed columns (an unknown column is an error)."""

    COLUMNS: ClassVar[list[str]] = ["time", "run", "split", "n", "cosine", "jaccard", "token_overlap", "bleu1", "rouge_l",
               "law_hit", "errors", "git", "note"]

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else RUNS / "results.tsv"

    def append(self, row: dict) -> None:
        unknown = set(row) - set(self.COLUMNS)
        if unknown:
            raise ValueError(f"unknown ledger columns {sorted(unknown)}")
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            if new:
                f.write("\t".join(self.COLUMNS) + "\n")
            f.write("\t".join(self._cell(row.get(c, "")) for c in self.COLUMNS) + "\n")

    def merge(self, other: Path) -> int:
        """Append rows of another ledger (e.g. pulled from the box) that are not here yet. Returns rows added."""
        have = set(self.path.read_text(encoding="utf-8").splitlines()) if self.path.exists() else set()
        lines = Path(other).read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].split("\t") != self.COLUMNS:
            raise ValueError(f"{other}: header differs from Ledger.COLUMNS")
        new = [ln for ln in lines[1:] if ln and ln not in have]
        if new:
            first = not self.path.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(("\t".join(self.COLUMNS) + "\n" if first else "") + "\n".join(new) + "\n")
        return len(new)

    @staticmethod
    def _cell(v) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v).replace("\t", " ").replace("\n", " ")


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip()
        return sha + ("-dirty" if dirty else "") if sha else "unknown"
    except FileNotFoundError:
        return "unknown"
