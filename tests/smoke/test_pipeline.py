"""Whole pipeline on real data (5 questions, 300 law files, BM25, fake LLM). Skipped until `make data` has run."""
from pathlib import Path

import pytest

from config import load
from pipeline import Experiment

pytestmark = pytest.mark.skipif(not Path("data/processed/qa.parquet").exists(), reason="run `make data` first")


def test_smoke_run_writes_predictions_and_metrics(tmp_path, monkeypatch):
    import runs

    monkeypatch.setattr(runs, "RUNS", tmp_path)  # keep the real data/runs/ and ledger clean
    summary = Experiment(load("configs/exp/S0_smoke.yaml")).execute()
    assert summary["n"] == 5 and summary["errors"] == 0
    assert 0 <= summary["rouge_l"] <= 1
    assert (tmp_path / "results.tsv").exists()
