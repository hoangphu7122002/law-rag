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


def test_trajectories_are_written_per_qid_and_duplicate_texts_share_a_trace(tmp_path):
    import json

    from pipeline import save_trajectories
    from records import Question

    class Agent:
        traces = {"q a": [{"role": "user", "content": "q a"}], "q b": [{"role": "user", "content": "q b"}]}

    qs = [Question(1, "q a", ""), Question(2, "q b", ""), Question(3, "q a", ""), Question(4, "unanswered", "")]
    run = Run(tmp_path)
    save_trajectories(run, qs, Agent())
    rows = [json.loads(ln) for ln in (tmp_path / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["qid"], r["question"]) for r in rows] == [(1, "q a"), (2, "q b"), (3, "q a")]
    assert rows[2]["messages"] == [{"role": "user", "content": "q a"}]


def test_no_trajectories_file_for_generators_without_traces(tmp_path):
    from pipeline import save_trajectories
    from records import Question

    class Plain:
        pass

    class Empty:
        traces: dict = {}

    for g in (Plain(), Empty()):
        save_trajectories(Run(tmp_path), [Question(1, "q", "")], g)
    assert not (tmp_path / "trajectories.jsonl").exists()
