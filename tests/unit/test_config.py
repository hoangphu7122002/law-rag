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


def test_tools_url_does_not_change_the_run_key():
    from config import load

    a, b = load("configs/exp/A7_react.yaml"), load("configs/exp/A7_react.yaml", ["generate.tools_url=http://localhost:8100"])
    assert a.run_key() == b.run_key()


def test_new_chunk_options_leave_existing_index_keys_unchanged_and_new_ones_distinct():
    base = load("configs/exp/S1_serve.yaml", backend="fake")
    assert base.chunk.split == "recursive"
    assert base.index_key() == "4ac2a75c"                                     # default split adds nothing to the hash
    sent = load("configs/exp/S1_serve.yaml", overrides=["chunk.split=sentence"], backend="fake")
    assert sent.index_key() != base.index_key()
    assert load("configs/exp/S1_serve.yaml", overrides=["chunk.split=sentence", "chunk.size=512"], backend="fake").index_key() != sent.index_key()
    sem = load("configs/exp/S1_serve.yaml", overrides=["chunk.split=semantic", "chunk.semantic_pct=95"], backend="fake")
    assert sem.index_key() != load("configs/exp/S1_serve.yaml", overrides=["chunk.split=semantic"], backend="fake").index_key()
