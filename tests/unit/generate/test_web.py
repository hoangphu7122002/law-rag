from generate.web import WebSearch


class FakeProvider:
    """Stands in for the external service (the only thing we fake)."""

    def __init__(self, results=None, fail=None):
        self.results, self.fail, self.calls = results or [], fail, []

    def search(self, query):
        self.calls.append(query)
        if self.fail:
            raise self.fail
        return self.results


R = lambda url, title="t", snippet="s": {"url": url, "title": title, "snippet": snippet}


def test_the_same_query_is_sent_to_the_service_only_once(tmp_path):
    p = FakeProvider([R("https://vbpl.vn/a")])
    w = WebSearch(p, cache=tmp_path / "web.sqlite")
    first, second = w.search("điều 5 nghị định"), w.search("điều 5 nghị định")
    assert first == second and p.calls == ["điều 5 nghị định"]


def test_the_cache_survives_a_new_instance_so_a_rerun_is_reproducible(tmp_path):
    WebSearch(FakeProvider([R("https://vbpl.vn/a")]), cache=tmp_path / "web.sqlite").search("q")
    p2 = FakeProvider([R("https://other.example/zzz")])
    assert WebSearch(p2, cache=tmp_path / "web.sqlite").search("q")[0]["url"] == "https://vbpl.vn/a" and p2.calls == []


def test_only_allowed_domains_survive_including_subdomains_and_not_lookalikes(tmp_path):
    p = FakeProvider([R("https://vbpl.vn/a"), R("https://www.vbpl.vn/b"), R("https://evil-vbpl.vn/c"), R("https://blog.example/d")])
    w = WebSearch(p, cache=tmp_path / "w.sqlite", domains=["vbpl.vn"])
    assert [r["url"] for r in w.search("q")] == ["https://vbpl.vn/a", "https://www.vbpl.vn/b"]


def test_no_domain_list_means_no_filtering_and_results_are_capped(tmp_path):
    p = FakeProvider([R(f"https://x{i}.example/") for i in range(9)])
    assert len(WebSearch(p, cache=tmp_path / "w.sqlite", max_results=4).search("q")) == 4


LEAK = "căn cứ điều 5 nghị định này quy định về việc quản lý và sử dụng nguồn kinh phí của các đơn vị trực thuộc"   # >= 13 words


def test_results_sharing_a_13_gram_with_a_blocked_text_are_dropped_to_protect_the_evaluation(tmp_path):
    p = FakeProvider([R("https://vbpl.vn/a", snippet=f"mở đầu {LEAK} kết thúc"), R("https://vbpl.vn/b", snippet="một đoạn khác hẳn không liên quan")])
    w = WebSearch(p, cache=tmp_path / "w.sqlite", blocked_texts=[LEAK])
    assert [r["url"] for r in w.search("q")] == ["https://vbpl.vn/b"]


def test_a_short_overlap_is_not_a_leak(tmp_path):
    p = FakeProvider([R("https://vbpl.vn/a", snippet="căn cứ điều 5 nghị định này quy định")])
    assert len(WebSearch(p, cache=tmp_path / "w.sqlite", blocked_texts=[LEAK]).search("q")) == 1


def test_tinyfish_provider_sends_the_key_as_a_header_and_normalises_results():
    from generate.web import TinyFish

    seen = {}

    def http(url, params, headers):
        seen.update(url=url, params=params, headers=headers)
        return 200, {"results": [{"position": 1, "site_name": "vbpl", "title": "T", "snippet": "S", "url": "https://vbpl.vn/a"}]}

    out = TinyFish(api_key="K", http=http).search("điều 5")
    assert out == [{"url": "https://vbpl.vn/a", "title": "T", "snippet": "S"}]
    assert seen["url"] == "https://api.search.tinyfish.ai" and seen["params"]["query"] == "điều 5" and seen["headers"]["X-API-Key"] == "K"


def test_tinyfish_without_a_key_fails_loudly_and_never_calls_out(monkeypatch):
    import pytest

    from generate.web import TinyFish

    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    called = []
    with pytest.raises(RuntimeError, match="TINYFISH_API_KEY"):
        TinyFish(http=lambda *a: called.append(a)).search("q")
    assert called == []


def test_tinyfish_http_error_is_reported_with_the_status():
    import pytest

    from generate.web import TinyFish

    with pytest.raises(RuntimeError, match="429"):
        TinyFish(api_key="K", http=lambda *a: (429, {})).search("q")


def test_web_settings_come_from_the_generate_config_and_default_to_off(tmp_path):
    from config import Generate
    from generate.react import LawTools

    g = Generate(model="m", params_b=2.0, agent=True)
    assert "web_search" not in g.agent_tools and g.web.provider == "tinyfish" and g.web.block_test_answers is True

    g = Generate(model="m", params_b=2.0, agent=True, agent_tools=["web_search"],
                 web={"domains": ["vbpl.vn"], "max_results": 3, "cache": str(tmp_path / "w.sqlite")})
    w = LawTools(g, provider=FakeProvider([R(f"https://vbpl.vn/{i}") for i in range(6)] + [R("https://blog.example/x")]), blocked_texts=[LEAK]).web
    assert [r["url"] for r in w.search("q")] == [f"https://vbpl.vn/{i}" for i in range(3)]


def test_uncached_calls_are_spaced_to_respect_the_rate_limit_but_cached_ones_are_not(tmp_path):
    now, slept = [0.0], []

    def sleep(s):
        slept.append(round(s, 3))
        now[0] += s

    w = WebSearch(FakeProvider([R("https://vbpl.vn/a")]), cache=tmp_path / "w.sqlite", calls_per_minute=30, clock=lambda: now[0], sleep=sleep)
    w.search("q1")
    w.search("q2")            # second paid call right away: must wait 60/30 = 2 s
    w.search("q1")            # cached: no call, no wait
    assert slept == [2.0]


def test_search_is_safe_to_call_from_several_threads_and_pays_once_per_query(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    p = FakeProvider([R("https://vbpl.vn/a")])
    w = WebSearch(p, cache=tmp_path / "w.sqlite", calls_per_minute=6000, sleep=lambda s: None)
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda i: w.search(f"q{i % 3}"), range(24)))
    assert all(len(r) == 1 for r in out) and sorted(p.calls) == ["q0", "q1", "q2"]


def test_the_hourly_limit_stops_paid_calls_with_a_clear_error_and_cached_queries_still_work(tmp_path):
    import pytest

    now = [0.0]
    p = FakeProvider([R("https://vbpl.vn/a")])
    w = WebSearch(p, cache=tmp_path / "w.sqlite", calls_per_minute=6000, calls_per_hour=3, clock=lambda: now[0], sleep=lambda s: None)
    for q in ("a", "b", "c"):
        w.search(q)
    with pytest.raises(RuntimeError, match="hourly"):
        w.search("d")
    assert w.search("a")[0]["url"] == "https://vbpl.vn/a"          # cached: free and allowed
    now[0] += 3601
    assert w.search("d")                                           # the window moved on


def test_rate_limits_come_from_the_web_config_so_a_higher_tier_needs_no_code_change(tmp_path):
    from config import Generate
    from generate.react import LawTools

    d = Generate(model="m", params_b=2.0, agent=True)
    assert (d.web.calls_per_minute, d.web.calls_per_hour, d.web.cost_per_query, d.web.max_cost_usd) == (30, 500, 0.0, None)   # Hobby: free
    g = Generate(model="m", params_b=2.0, agent=True, web={"calls_per_minute": 150, "calls_per_hour": 5000, "cache": str(tmp_path / "w.sqlite")})
    w = LawTools(g, provider=FakeProvider(), blocked_texts=[]).web
    assert (w.per_hour, round(w.gap, 3)) == (5000, 0.4)


def test_spending_is_capped_across_runs_and_cached_queries_cost_nothing(tmp_path):
    import pytest

    cache = tmp_path / "w.sqlite"
    mk = lambda: WebSearch(FakeProvider([R("https://vbpl.vn/a")]), cache=cache, calls_per_minute=6000, sleep=lambda s: None,
                           cost_per_query=0.005, max_cost_usd=0.012)
    w = mk()
    w.search("a"), w.search("b")                                    # $0.010 spent
    assert round(w.spent(), 3) == 0.010
    w.search("a")                                                   # cached: still $0.010
    assert round(w.spent(), 3) == 0.010
    with pytest.raises(RuntimeError, match="budget"):
        w.search("c")                                               # would reach $0.015 > cap $0.012
    again = mk()                                                    # a new process sees the same spend
    assert round(again.spent(), 3) == 0.010
    with pytest.raises(RuntimeError, match="budget"):
        again.search("d")


def test_tinyfish_reads_the_key_from_a_file_so_it_is_never_on_a_command_line(tmp_path, monkeypatch):
    from generate.web import TinyFish

    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    f = tmp_path / "tinyfish.key"
    f.write_text("sk-file-key\n")
    seen = {}
    TinyFish(key_file=str(f), http=lambda url, params, headers: (seen.update(h=headers) or (200, {"results": []}))).search("q")
    assert seen["h"]["X-API-Key"] == "sk-file-key"


def test_the_environment_key_wins_over_the_file(tmp_path, monkeypatch):
    from generate.web import TinyFish

    monkeypatch.setenv("TINYFISH_API_KEY", "sk-env")
    f = tmp_path / "k"
    f.write_text("sk-file")
    seen = {}
    TinyFish(key_file=str(f), http=lambda url, params, headers: (seen.update(h=headers) or (200, {"results": []}))).search("q")
    assert seen["h"]["X-API-Key"] == "sk-env"


def test_web_key_file_setting_reaches_the_provider(tmp_path, monkeypatch):
    from config import Generate
    from generate.react import LawTools

    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    f = tmp_path / "k"
    f.write_text("sk-cfg")
    g = Generate(model="m", params_b=2.0, agent=True, web={"key_file": str(f), "cache": str(tmp_path / "w.sqlite")})
    assert LawTools(g, blocked_texts=[]).web.provider.key_file == str(f)
