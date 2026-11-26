import json
import sys
from pathlib import Path

from subset_sft import subset_rows, write_subset


def rows(n_q=100, per=2):
    return [{"qid": q, "prompt": f"p{q}-{i}", "completion": "c"} for q in range(n_q) for i in range(per)]


def test_subset_keeps_whole_questions_and_the_requested_share():
    out = subset_rows(rows(), fraction=0.25, seed=1)
    qids = {r["qid"] for r in out}
    assert len(qids) == 25 and len(out) == 50                      # both rows of every kept question
    assert all(sum(1 for r in out if r["qid"] == q) == 2 for q in qids)


def test_subset_is_reproducible_and_depends_on_the_seed():
    a = subset_rows(rows(), 0.5, seed=1)
    assert a == subset_rows(rows(), 0.5, seed=1)
    assert {r["qid"] for r in a} != {r["qid"] for r in subset_rows(rows(), 0.5, seed=2)}


def test_full_fraction_is_identity_and_smaller_is_nested_in_larger():
    assert subset_rows(rows(), 1.0, seed=3) == rows()
    small = {r["qid"] for r in subset_rows(rows(), 0.25, seed=3)}
    big = {r["qid"] for r in subset_rows(rows(), 0.5, seed=3)}
    assert small <= big


def test_write_subset_copies_val_unchanged(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows(20)) + "\n")
    (src / "val.jsonl").write_text('{"qid": 999, "prompt": "v", "completion": "c"}\n')
    out = tmp_path / "out"
    n = write_subset(src, out, fraction=0.5, seed=1)
    assert n == 20 and (out / "val.jsonl").read_text() == (src / "val.jsonl").read_text()
    assert len((out / "train.jsonl").read_text().splitlines()) == 20
