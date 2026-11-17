import pytest
from pydantic import ValidationError

from config import load


def test_experiment_overrides_base_and_set_overrides_experiment():
    cfg = load("configs/exp/B0_baseline.yaml", overrides=["retrieve.k=10"], backend="fake")
    assert cfg.name == "B0_baseline"
    assert cfg.embed.model == "keepitreal/vietnamese-sbert"   # from the experiment
    assert cfg.rerank.model is None                          # from base
    assert cfg.retrieve.k == 10                              # from --set
    assert cfg.backend.name == "fake"


def test_unknown_key_is_an_error():
    with pytest.raises(ValidationError):
        load("configs/exp/B0_baseline.yaml", overrides=["retrieve.top_k=3"], backend="fake")


def test_answering_llm_must_be_under_4b():
    with pytest.raises(ValidationError, match="< 4B"):
        load("configs/exp/B0_baseline.yaml", overrides=["generate.params_b=4.0"], backend="fake")


def test_changing_the_llm_keeps_the_retrieval_cache_key():
    a = load("configs/exp/B0_baseline.yaml", backend="fake")
    b = load("configs/exp/B0_baseline.yaml", overrides=["generate.max_tokens=768"], backend="fake")
    assert a.retrieval_key() == b.retrieval_key()
    assert a.run_key() != b.run_key()
