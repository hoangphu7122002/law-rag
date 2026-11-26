import pytest

from train.weights.wsp_env import Budget


def test_budget_counts_down_persists_and_refuses_after_the_limit(tmp_path):
    b = Budget(tmp_path / "b.txt", 2)
    assert b.spend() == 1
    assert Budget(tmp_path / "b.txt", 2).used == 1        # persisted across processes
    assert b.spend() == 0
    with pytest.raises(RuntimeError):
        b.spend()
