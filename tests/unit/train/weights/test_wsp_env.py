import pytest

from train.weights.wsp_env import Budget


def test_budget_counts_down_persists_and_refuses_after_the_limit(tmp_path):
    b = Budget(tmp_path / "b.txt", 2)
    assert b.spend() == 1
    assert Budget(tmp_path / "b.txt", 2).used == 1        # persisted across processes
    assert b.spend() == 0
    with pytest.raises(RuntimeError):
        b.spend()


def test_paired_se_is_stdev_over_sqrt_n_and_nan_for_one_item():
    import math

    from train.weights.wsp_env import paired_se
    assert math.isclose(paired_se([1.0, 3.0]), math.sqrt(2) / math.sqrt(2))
    assert math.isnan(paired_se([1.0]))
