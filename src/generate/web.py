"""Web search tool for the ReAct agent (config `generate.agent_tools: [..., web_search]`; off by default).

Design (docs/notes/15-web-search.md): a provider (TinyFish Search API) behind a tiny interface, a persistent query cache
(reproducible reruns, one paid call per query), a domain allow-list, a leak filter and size caps.
"""
import json
import os
import re
import time
import sqlite3
import threading
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urlparse

from config import DEFAULT_DENY_PATHS   # re-exported: the default deny-list lives next to WebCfg
from corpus.decontam import grams

_LATE_YEAR = re.compile(r"(?<!\d)(20(?:2[6-9]|[3-9]\d))(?!\d)")   # a year >= 2026


class WebSearch:
    def __init__(self, provider, cache: str | Path = ":memory:", domains: list[str] | None = None, max_results: int = 5, deny_paths: list[str] | None = None,
                 blocked_texts: list[str] | None = None, calls_per_minute: int = 30, calls_per_hour: int = 500, cost_per_query: float = 0.0, max_cost_usd: float | None = None, clock=time.monotonic, sleep=time.sleep):
        self.gap, self.clock, self.sleep, self._last = 60.0 / calls_per_minute, clock, sleep, None   # TinyFish pay-as-you-go: 30 searches / min,
        self.per_hour, self._hour = calls_per_hour, []                                            # 500 / hour (both free)
        self.provider, self.domains, self.max_results = provider, [d.lower() for d in domains or []], max_results
        self.deny = [re.compile(p, re.I) for p in (DEFAULT_DENY_PATHS if deny_paths is None else deny_paths)]
        self.blocked = set().union(*(grams(t) for t in blocked_texts or []))   # 13-grams of texts that must never come back (test answers)
        if str(cache) != ":memory:":
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
        self.db, self.lock = sqlite3.connect(str(cache), check_same_thread=False), threading.Lock()   # tools run in worker threads
        self.db.execute("CREATE TABLE IF NOT EXISTS q(query TEXT PRIMARY KEY, results TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS spend(calls INTEGER)")
        self.cost, self.max_cost = cost_per_query, max_cost_usd

    def search(self, query: str) -> list[dict]:
        with self.lock:   # one thread at a time: a query is paid once and the rate limit holds
            return self._search(query)

    def _search(self, query: str) -> list[dict]:
        row = self.db.execute("SELECT results FROM q WHERE query=?", (query,)).fetchone()
        if row:
            raw = json.loads(row[0])
        else:
            self._pace()
            raw = self.provider.search(query)
            self.db.execute("INSERT OR REPLACE INTO q VALUES(?,?)", (query, json.dumps(raw, ensure_ascii=False)))
            self.db.execute("INSERT INTO spend VALUES(1)")
            self.db.commit()
        out = [dict(r) for r in raw if self.allowed(r.get("url", "")) and not self.leaks(r)][: self.max_results]
        for r in out:
            if _LATE_YEAR.search(f"{r.get('url', '')} {r.get('title', '')}"):
                r["post_2025"] = True   # web returns the newest consolidated versions; references were written in 2025
        return out   # the cache keeps the raw answer; filters are applied on read

    def spent(self) -> float:
        """USD spent so far over every run that shares this cache (one row per paid call)."""
        return self.db.execute("SELECT COUNT(*) FROM spend").fetchone()[0] * self.cost

    def _pace(self) -> None:
        if self.max_cost is not None and self.spent() + self.cost > self.max_cost + 1e-9:
            raise RuntimeError(f"web search budget reached (${self.spent():.3f} of ${self.max_cost:.3f}); cached queries still work")
        now = self.clock()
        self._hour = [t for t in self._hour if now - t < 3600]
        if len(self._hour) >= self.per_hour:
            raise RuntimeError(f"web search hourly limit reached ({self.per_hour} calls/hour); cached queries still work, retry later")
        self._hour.append(now)
        if self._last is not None and self.clock() - self._last < self.gap:
            self.sleep(self.gap - (self.clock() - self._last))
        self._last = self.clock()

    def leaks(self, r: dict) -> bool:
        return bool(self.blocked) and bool(grams(f"{r.get('title', '')} {r.get('snippet', '')}") & self.blocked)

    def allowed(self, url: str) -> bool:
        u = urlparse(url)
        if self.domains and not any((u.hostname or "") == d or (u.hostname or "").endswith("." + d) for d in self.domains):
            return False
        target = urllib.parse.unquote(u.path + ("?" + u.query if u.query else ""))
        return not any(p.search(target) for p in self.deny)


class TinyFish:
    """TinyFish Search API (https://docs.tinyfish.ai/search-api): GET with the key in `X-API-Key`; key from TINYFISH_API_KEY (never in a file).
    Response shape verified with one live call on 2026-11-19: {query, results[{position, site_name, snippet, title, url}], total_results, page}, 10 results per call."""
    URL = "https://api.search.tinyfish.ai"

    def __init__(self, api_key: str | None = None, http=None, key_file: str | None = None):
        self.api_key, self.http, self.key_file = api_key, http or self._get, key_file

    def search(self, query: str) -> list[dict]:
        key = self.api_key or os.environ.get("TINYFISH_API_KEY", "")
        if not key and self.key_file and Path(self.key_file).is_file():
            key = Path(self.key_file).read_text().strip()
        if not key:
            raise RuntimeError("TINYFISH_API_KEY is not set (and no readable web.key_file)")
        status, data = self.http(self.URL, {"query": query}, {"X-API-Key": key})
        if status != 200:
            raise RuntimeError(f"TinyFish search failed: HTTP {status}")
        return [{"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("snippet", "")} for r in data.get("results", [])]

    @staticmethod
    def _get(url: str, params: dict, headers: dict) -> tuple[int, dict]:
        req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, {}
