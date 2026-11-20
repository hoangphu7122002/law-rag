import threading
from types import SimpleNamespace as NS

from generate.tools_service import RemoteTools, make_server
from records import Passage


class FakeTools:
    def __init__(self):
        self.calls = []
        self.web = NS(search=lambda q: [{"title": "t", "url": "u", "snippet": q}])

    def search_law(self, q):
        self.calls.append(("search", q))
        return [Passage("p1", "50_2024_TT", "Điều 5. Nội dung", 0.9)]

    def article(self, law, dieu):
        self.calls.append(("article", law, dieu))
        return "Điều 5. Toàn văn"

    def related(self, law, hops):
        self.calls.append(("related", law, hops))
        return [{"law": "x"}]


def start(tools):
    srv = make_server(tools, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_remote_tools_round_trip_every_tool():
    tools = FakeTools()
    srv, url = start(tools)
    try:
        r = RemoteTools(url)
        ps = r.search_law("mức phạt")
        assert ps == [Passage("p1", "50_2024_TT", "Điều 5. Nội dung", 0.9)]
        assert r.article("50/2024/TT", 5) == "Điều 5. Toàn văn"
        assert r.related("75/2015/QH13", 2) == [{"law": "x"}]
        assert r.web.search("luật")[0]["snippet"] == "luật"
    finally:
        srv.shutdown()


def test_same_search_is_served_from_cache_so_the_index_is_hit_once():
    tools = FakeTools()
    srv, url = start(tools)
    try:
        r = RemoteTools(url)
        r.search_law("q"); r.search_law("q"); r.search_law("q2")
        assert tools.calls.count(("search", "q")) == 1 and ("search", "q2") in tools.calls
    finally:
        srv.shutdown()


def test_unknown_article_is_an_empty_string_not_an_error():
    class T(FakeTools):
        def article(self, law, dieu):
            return ""
    srv, url = start(T())
    try:
        assert RemoteTools(url).article("x", 1) == ""
    finally:
        srv.shutdown()
