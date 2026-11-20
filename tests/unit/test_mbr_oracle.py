import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from mbr_oracle import summarize


def test_best_of_n_is_at_least_greedy_and_mbr():
    rows = [{"ref": "a b c d", "cands": ["a b x y", "a b c d", "a b c z", "q r s t"]},
            {"ref": "m n o p", "cands": ["m n o p", "m n o p", "x y z w", "x y z w"]}]
    s = summarize(rows)
    assert s["best_of_4"] >= s["mbr"] >= 0 and s["best_of_4"] >= s["greedy"]
    assert s["best_of_4"] == 1.0 and s["n"] == 2
