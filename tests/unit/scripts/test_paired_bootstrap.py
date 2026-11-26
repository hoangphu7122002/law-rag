from paired_bootstrap import paired_bootstrap


def test_a_consistently_better_model_has_a_confidence_interval_above_zero():
    a = {i: 0.7 for i in range(100)}
    b = {i: 0.6 + (i % 3) * 0.01 for i in range(100)}
    d, lo, hi = paired_bootstrap(a, b, n=500, seed=0)
    assert d > 0.09 and lo > 0.08 and hi >= d


def test_identical_models_have_zero_difference_and_an_interval_around_zero():
    a = {i: (i % 10) / 10 for i in range(100)}
    d, lo, hi = paired_bootstrap(a, dict(a), n=200, seed=0)
    assert d == 0 and lo == 0 and hi == 0


def test_only_questions_scored_by_both_models_are_used():
    a = {1: 1.0, 2: 1.0, 3: 0.0}
    b = {2: 0.0, 3: 0.0, 4: 1.0}
    d, _, _ = paired_bootstrap(a, b, n=100, seed=0)
    assert abs(d - 0.5) < 1e-9
