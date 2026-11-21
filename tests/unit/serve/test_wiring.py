from types import SimpleNamespace

from serve.wiring import overrides_from_env, probe


def test_env_urls_become_config_overrides():
    env = {"LLM_URL": "http://llm:8000/v1", "EMBED_URL": "http://embed:8100/v1", "RERANK_URL": "http://rerank:8200", "OTHER": "x"}
    assert overrides_from_env(env) == ["backend.base_url=http://llm:8000/v1", "backend.embed_url=http://embed:8100/v1",
                                       "backend.rerank_url=http://rerank:8200"]
    assert overrides_from_env({}) == []


def test_probe_asks_the_server_root_health_of_each_endpoint():
    seen = []

    def get(url, timeout):
        seen.append(url)
        return SimpleNamespace(status_code=200 if "llm" in url else 500)

    out = probe({"llm": "http://llm:8000/v1", "rerank": "http://rerank:8200"}, get=get)
    assert out == {"llm": True, "rerank": False}
    assert seen == ["http://llm:8000/health", "http://rerank:8200/health"]          # /v1 stripped: vLLM serves /health at the root


def test_probe_treats_a_connection_error_as_down():
    def get(url, timeout):
        raise ConnectionError("refused")

    assert probe({"llm": "http://llm:8000/v1"}, get=get) == {"llm": False}


def test_limiter_from_env_treats_unset_and_empty_as_unlimited():
    from serve.wiring import limiter_from_env

    for env in ({}, {"MAX_CONCURRENT": "", "MAX_QUEUE": ""}):
        lim = limiter_from_env(env)
        assert lim.max_concurrent is None and lim.max_queue is None
    lim = limiter_from_env({"MAX_CONCURRENT": "8", "MAX_QUEUE": "16"})
    assert (lim.max_concurrent, lim.max_queue) == (8, 16)
