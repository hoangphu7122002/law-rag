"""Extra corpus from luatvietnam.vn for cited documents that neither the course corpus nor vbpl has
(internal NHNN / KBNN / BHTG decisions and notices, the newest laws). See citations.py.

Polite crawl: robots.txt disallows the search pages, so documents are found through the public sitemaps
(slug contains the number, e.g. ".../quyet-dinh-2345-qd-nhnn-2023-...-285123-d1.html"); one request per DELAY s.
Output mirrors vbpl: data/processed/web/<number>__<page id>.txt (first line = title) + manifest.parquet.

    LuatVietnam().build(refs)        # refs: rows of Citations().table() with where == "web"
"""
import html
import re
import time
import unicodedata
from pathlib import Path
from typing import ClassVar

from corpus.sources.vbpl import VbplImporter

RAW = Path("data/raw/luatvietnam")
OUT = Path("data/processed/web")
BASE = "https://luatvietnam.vn"


def slug(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("Đ", "D").replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


class LuatVietnam:
    UA: ClassVar[str] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
    DELAY: ClassVar[float] = 1.0
    MAX_HITS: ClassVar[int] = 3     # same number in several years -> keep the newest few

    def __init__(self):
        import requests

        self.http = requests.Session()
        self.http.headers["User-Agent"] = self.UA
        self._last = 0.0

    def get(self, url: str) -> str:
        time.sleep(max(0.0, self.DELAY - (time.time() - self._last)))
        self._last = time.time()
        r = self.http.get(url, timeout=60)
        r.raise_for_status()
        return r.text

    # ---- find --------------------------------------------------------------------------------------------------
    def urls(self) -> list[str]:
        """All document URLs from the sitemaps, cached in data/raw/luatvietnam/urls.txt."""
        cache = RAW / "urls.txt"
        if cache.exists():
            return cache.read_text().split()
        RAW.mkdir(parents=True, exist_ok=True)
        maps = re.findall(r"<loc>([^<]+)</loc>", self.get(f"{BASE}/sitemap.xml"))
        maps = [m for m in maps if re.search(r"sitemap_(?:law)?document\d+\.xml$", m)]
        out = []
        for i, m in enumerate(maps):
            out += re.findall(r"<loc>([^<]+-d\d\.html)</loc>", self.get(m))
            if i % 50 == 0:
                print(f"sitemap {i}/{len(maps)}: {len(out)} urls")
        out = sorted(set(out))
        cache.write_text("\n".join(out))
        return out

    def match(self, ref: dict, urls: list[str]) -> list[str]:
        """Candidate URLs for a citation: number in the slug (newest first), or law name + year."""
        if ref["kind"] == "number":
            pat = re.compile(rf"/[a-z-]*?-{re.escape(slug(ref['number']))}(?:-[a-z0-9-]*)?-(\d+)-d\d\.html$")
        else:
            name = slug(re.sub(r"\s+\d{4}$", "", ref["ref"]))
            pat = re.compile(rf"/{re.escape(name)}-(?:[a-z0-9-]*-)?{ref['year']}-[a-z0-9-]*?(\d+)-d\d\.html$")
        hits = [(int(m.group(1)), u) for u in urls if (m := pat.search(u))]
        return [u for _, u in sorted(hits, reverse=True)[: self.MAX_HITS]]

    @staticmethod
    def page_id(url: str) -> str:
        return re.search(r"-(\d+)-d\d\.html$", url).group(1)

    # ---- fetch -------------------------------------------------------------------------------------------------
    def document(self, url: str) -> tuple[str, str]:
        """(title, text) of a document page; text = the full-text body only."""
        cache = RAW / "pages" / f"{self.page_id(url)}.html"
        if cache.exists():
            page = cache.read_text(encoding="utf-8")
        else:
            page = self.get(url)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(page, encoding="utf-8")
        title = re.search(r'class="the-document-title[^"]*"[^>]*>(.*?)</', page, re.DOTALL)
        start = page.find('class="the-document-body"')
        end = page.rfind("<", start, page.find("the-document-vbl", start))  # cut before the tag, not inside it
        body = page[start: end if end > start else None] if start >= 0 else ""
        title = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title.group(1)))).strip() if title else ""
        return title, VbplImporter.to_text(body.split(">", 1)[-1])

    def build(self, refs: list[dict]) -> int:
        import polars as pl

        urls, rows, done = self.urls(), [], set()
        OUT.mkdir(parents=True, exist_ok=True)
        for ref in refs:
            for url in self.match(ref, urls):
                if url in done:
                    continue
                done.add(url)
                try:
                    title, text = self.document(url)
                except Exception as e:  # noqa: BLE001 - one bad page must not stop the crawl
                    print(f"skip {url}: {e}")
                    continue
                if len(text) < 200:
                    continue
                num = re.search(r"Số:\s*([\w/Đđ.-]+)", text)
                name = f"{VbplImporter.filename(num.group(1)) if num else 'doc'}__{self.page_id(url)}"  # same number, other years
                (OUT / f"{name}.txt").write_text(f"{title}\n\n{text}", encoding="utf-8")
                rows.append({"file": f"{name}.txt", "ref": ref["ref"], "url": url, "title": title,
                             "so_ky_hieu": num.group(1) if num else None})
        pl.DataFrame(rows).write_parquet(OUT / "manifest.parquet")
        print(f"web: {len(refs)} refs -> {len(rows)} documents in {OUT}")
        return len(rows)
