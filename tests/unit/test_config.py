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


def test_tool_set_and_web_settings_can_be_overridden_from_the_command_line():
    from config import load

    c = load("configs/exp/A7_react.yaml", ['generate.agent_tools=["get_article","kg_related"]', 'generate.web.domains=["vbpl.vn"]', "split=dev_clean"])
    assert c.generate.agent_tools == ["get_article", "kg_related"] and c.generate.web.domains == ["vbpl.vn"] and c.split == "dev_clean"


def test_new_agent_fields_at_their_defaults_do_not_change_old_run_keys():
    from config import load

    a = load("configs/exp/A4_q35_off.yaml").run_key()
    b = load("configs/exp/A4_q35_off.yaml", ["generate.agent_tools=[\"search_law\",\"get_article\"]"]).run_key()
    assert a == b


def test_agent_seed_changes_run_key_only_when_set():
    from config import load

    base = load("configs/exp/A4_q35_off.yaml").run_key()
    assert load("configs/exp/A4_q35_off.yaml", ["generate.agent_seed=0"]).run_key() == base
    k1 = load("configs/exp/A4_q35_off.yaml", ["generate.agent_seed=1"]).run_key()
    k2 = load("configs/exp/A4_q35_off.yaml", ["generate.agent_seed=2"]).run_key()
    assert len({base, k1, k2}) == 3


def test_oracle_keeps_retrieval_key_but_changes_run_key():
    a = load("configs/exp/B0_baseline.yaml", backend="fake")
    b = load("configs/exp/B0_baseline.yaml", ["retrieve.oracle=true"], backend="fake")
    assert a.retrieval_key() == b.retrieval_key()
    assert a.run_key() != b.run_key()


def test_oracle_off_leaves_run_key_unchanged():
    a = load("configs/exp/B0_baseline.yaml", backend="fake")
    b = load("configs/exp/B0_baseline.yaml", ["retrieve.oracle=false"], backend="fake")
    assert a.run_key() == b.run_key()
