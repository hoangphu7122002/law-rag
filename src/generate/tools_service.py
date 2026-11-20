"""One shared tools endpoint (like datamate's serve adapters): the index / docstore / reranker client / KG / web search live in ONE process,
every run (and every worker thread of a run) calls it over HTTP, so a 10-15 GB docstore is held once instead of once per run.

    PYTHONPATH=src python -m generate.tools_service --exp configs/exp/A7_react.yaml --port 8300
    ... --set generate.tools_url=http://localhost:8300       # the agent then uses RemoteTools instead of building its own index

Endpoints (POST JSON): /search_law {query} -> [passage]; /article {law, dieu} -> {text}; /related {law, hops} -> [..]|null; /web {query} -> [..].
search_law results are cached by query (trajectory sampling repeats the same queries a lot)."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from records import Passage


def make_server(tools, host: str = "127.0.0.1", port: int = 8300) -> ThreadingHTTPServer:
    cache: dict[str, list[dict]] = {}
    lock = threading.Lock()

    def search(q: str) -> list[dict]:
        with lock:                                   # one build / one lookup at a time; a cache hit is the common case
            if q not in cache:
                cache[q] = [vars(p) if hasattr(p, "__dict__") else p._asdict() for p in tools.search_law(q)]
            return cache[q]

    routes = {
        "/search_law": lambda a: search(str(a["query"])),
        "/article": lambda a: {"text": tools.article(a["law"], int(a["dieu"])) or ""},
        "/related": lambda a: tools.related(str(a["law"]), int(a.get("hops", 1))),
        "/web": lambda a: tools.web.search(str(a["query"])),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                a = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                body, code = json.dumps(routes[self.path](a), ensure_ascii=False).encode(), 200
            except Exception as e:  # noqa: BLE001 — the client turns it into a tool error text
                body, code = json.dumps({"error": repr(e)[:300]}).encode(), 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


class RemoteTools:
    """Same interface as generate.react.LawTools, talking to the shared endpoint."""

    def __init__(self, url: str, timeout: float = 300.0):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.web = SimpleNamespace(search=lambda q: self._post("/web", {"query": q}))

    def _post(self, path: str, payload: dict):
        import urllib.request

        req = urllib.request.Request(self.url + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def search_law(self, query: str) -> list[Passage]:
        return [Passage(**p) for p in self._post("/search_law", {"query": query})]

    def article(self, law: str, dieu: int) -> str:
        return self._post("/article", {"law": law, "dieu": dieu})["text"]

    def related(self, law: str, hops: int):
        return self._post("/related", {"law": law, "hops": hops})


def main() -> None:
    from config import load
    from generate.react import LawTools

    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--port", type=int, default=8300)
    a = p.parse_args()
    srv = make_server(LawTools(load(a.exp, a.set, None)), "0.0.0.0", a.port)
    print(f"tools endpoint on :{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
