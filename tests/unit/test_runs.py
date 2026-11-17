import pytest

from runs import Ledger, Run


def test_predictions_resume_and_skip_cut_lines(tmp_path):
    run = Run(tmp_path)
    run.append_prediction({"qid": 1, "answer": "a"})
    (tmp_path / "predictions.jsonl").open("a").write('{"qid": 2, "ans')  # crash mid-line
    assert run.done_qids() == {1}


def test_ledger_rejects_unknown_columns(tmp_path):
    ledger = Ledger(tmp_path / "results.tsv")
    ledger.append({"run": "x", "rouge_l": 0.25})
    with pytest.raises(ValueError):
        ledger.append({"run": "x", "rougeL": 0.25})
    header, row = (tmp_path / "results.tsv").read_text().splitlines()
    assert header.split("\t") == Ledger.COLUMNS and "0.2500" in row
