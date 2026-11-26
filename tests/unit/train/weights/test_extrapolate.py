import torch

from train.weights.extrapolate import extrapolate


def test_extrapolate_moves_along_the_base_to_tuned_direction_and_keeps_non_float_tensors():
    base = {"w": torch.tensor([1.0, 2.0]), "ids": torch.tensor([3, 4])}
    tuned = {"w": torch.tensor([2.0, 0.0]), "ids": torch.tensor([3, 4])}
    assert torch.allclose(extrapolate(base, tuned, 0.0)["w"], base["w"])
    assert torch.allclose(extrapolate(base, tuned, 1.0)["w"], tuned["w"])
    assert torch.allclose(extrapolate(base, tuned, 0.5)["w"], torch.tensor([1.5, 1.0]))
    assert torch.allclose(extrapolate(base, tuned, 2.0)["w"], torch.tensor([3.0, -2.0]))
    assert torch.equal(extrapolate(base, tuned, 2.0)["ids"], base["ids"])


def test_extrapolate_keeps_the_dtype_of_the_base_tensors():
    base = {"w": torch.ones(2, dtype=torch.bfloat16)}
    tuned = {"w": torch.zeros(2, dtype=torch.bfloat16)}
    assert extrapolate(base, tuned, 1.5)["w"].dtype == torch.bfloat16
