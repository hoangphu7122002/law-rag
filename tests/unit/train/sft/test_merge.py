import numpy as np
import pytest

from train.sft.merge import soup, task_arithmetic


def sd(a, b, step=1):
    return {"w": np.array(a, dtype=np.float32), "b": np.array(b, dtype=np.float32), "step": np.array([step])}


def test_uniform_soup_is_mean():
    out = soup([sd([0, 2], [4], 7), sd([2, 4], [8], 7)])
    assert np.allclose(out["w"], [1, 3]) and np.allclose(out["b"], [6])


def test_weighted_soup_normalises_weights():
    out = soup([sd([0], [0]), sd([4], [8])], weights=[1, 3])
    assert np.allclose(out["w"], [3]) and np.allclose(out["b"], [6])


def test_soup_keeps_int_entries_and_float16_dtype():
    a, b = sd([1], [1], 5), sd([3], [3], 9)
    a["w"], b["w"] = a["w"].astype(np.float16), b["w"].astype(np.float16)
    out = soup([a, b])
    assert out["w"].dtype == np.float16 and out["step"].tolist() == [5]


def test_soup_mismatched_keys_raises():
    with pytest.raises(ValueError):
        soup([sd([1], [1]), {"w": np.zeros(1)}])


def test_soup_does_not_mutate_inputs():
    a, b = sd([0, 2], [4]), sd([2, 4], [8])
    soup([a, b])
    assert a["w"].tolist() == [0, 2] and b["b"].tolist() == [8]


def test_arith_alpha_zero_returns_base():
    base, tuned, origin = sd([1, 1], [2]), sd([5, 5], [9]), sd([0, 0], [1])
    out = task_arithmetic(base, [(tuned, origin, 0.0)])
    assert np.allclose(out["w"], [1, 1]) and np.allclose(out["b"], [2])


def test_arith_alpha_one_with_base_equal_origin_returns_tuned():
    base = sd([1, 2], [3])
    tuned = sd([5, 7], [9])
    out = task_arithmetic(base, [(tuned, base, 1.0)])
    assert np.allclose(out["w"], tuned["w"]) and np.allclose(out["b"], tuned["b"])


def test_arith_sums_several_deltas_and_keeps_ints_and_dtype():
    base = sd([1], [1], 3)
    d1 = (sd([3], [1], 8), sd([1], [1], 8), 1.0)    # +2
    d2 = (sd([2], [5], 8), sd([0], [1], 8), 0.5)    # +1, +2
    base["w"] = base["w"].astype(np.float16)
    out = task_arithmetic(base, [d1, d2])
    assert out["w"].dtype == np.float16 and np.allclose(out["w"], [4]) and np.allclose(out["b"], [3])
    assert out["step"].tolist() == [3]


def test_arith_mismatched_keys_raises_and_inputs_unchanged():
    base, tuned, origin = sd([1], [1]), sd([2], [2]), sd([0], [0])
    with pytest.raises(ValueError):
        task_arithmetic(base, [({"w": np.zeros(1)}, origin, 1.0)])
    task_arithmetic(base, [(tuned, origin, 1.0)])
    assert base["w"].tolist() == [1] and tuned["w"].tolist() == [2] and origin["w"].tolist() == [0]
