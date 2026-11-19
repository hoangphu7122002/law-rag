"""Legal knowledge graph, built by RULES from vbpl.vn metadata + relationships (no LLM), stored in SQLite.

  nodes   one per document, primary key = vbpl id (the law number `key`, see corpus/keys.py, is NOT unique: 171k documents share
          70k keys, e.g. 84 provincial "01/2024/QĐ-UBND"), with title, issuer, validity status and dates
  edges   document -> document, `family` = amends | replaces | repeals | expires | guides | bases_on | cites | other,
          `raw` = the vbpl relationship name. vbpl lists each relation on BOTH documents under an active and a passive
          name (e.g. "Sửa đổi, bổ sung" on one side, "Văn bản bổ sung" on the other), so edges are kept as an UNDIRECTED
          pair per family (src < dst); the direction of each raw name is not verified yet (docs/notes/14-kg.md).
Operations are idempotent upserts / deletes by document id, so the graph can be updated without a rebuild:
    kg = KG("data/kg/kg.sqlite"); kg.upsert_doc(id, key=..., status=...); kg.add_edge(id_a, id_b, "amends"); kg.remove_doc(id)
Queries (by law key; every document carrying that key is included; documents without a law number are never matched):
    kg.neighbors(key, families=(...), as_of="2020-01-01"), kg.amendments(key), kg.ids(key)
"""
import re
import sqlite3
from datetime import date
from pathlib import Path

FAMILY = {
    "Sửa đổi, bổ sung": "amends", "Văn bản bổ sung": "amends", "Văn bản sửa đổi": "amends", "Văn bản được bổ sung": "amends",
    "Văn bản được sửa đổi": "amends", "Đính chính": "amends",
    "Thay thế": "replaces", "Hợp nhất": "replaces",
    "Bãi bỏ": "repeals",
    "Văn bản quy định hết hiệu lực": "expires", "Văn bản hết hiệu lực": "expires", "Văn bản quy định hết hiệu lực 1 phần": "expires",
    "Văn bản bị hết hiệu lực 1 phần": "expires", "Tạm ngưng hiệu lực": "expires", "Đình chỉ thi hành": "expires",
    "Văn bản đình chỉ": "expires", "Văn bản đình chỉ 1 phần": "expires", "Văn bản bị đình chỉ": "expires", "Văn bản bị đình chỉ 1 phần": "expires",
    "Quy định chi tiết, hướng dẫn thi hành": "guides", "Văn bản HD, QĐ chi tiết": "guides", "Văn bản được HD, QĐ chi tiết": "guides",
    "Hướng dẫn áp dụng": "guides", "Giải thích": "guides",
    "Căn cứ": "bases_on", "Văn bản căn cứ": "bases_on",
    "Dẫn chiếu": "cites", "Văn bản dẫn chiếu": "cites",
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, key TEXT, title TEXT, issuer TEXT, status TEXT, eff_from TEXT, eff_to TEXT);
CREATE INDEX IF NOT EXISTS docs_key ON docs(key);
CREATE TABLE IF NOT EXISTS edges(a TEXT, b TEXT, family TEXT, raw TEXT, PRIMARY KEY(a, b, family, raw));
CREATE INDEX IF NOT EXISTS edges_a ON edges(a); CREATE INDEX IF NOT EXISTS edges_b ON edges(b);
"""


def iso(d: str | None) -> str | None:
    """'dd/mm/yyyy' -> 'yyyy-mm-dd' (sortable), None if missing / unparsable."""
    m = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*", d or "")
    return f"{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}" if m else None


class KG:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    # ---- updates (idempotent) ----
    def upsert_doc(self, id: str, key: str = "", title: str = "", issuer: str = "", status: str = "", eff_from: str | None = None,
                   eff_to: str | None = None) -> None:
        self.db.execute("INSERT INTO docs VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET key=excluded.key, title=excluded.title, "
                        "issuer=excluded.issuer, status=excluded.status, eff_from=excluded.eff_from, eff_to=excluded.eff_to",
                        (id, key, title, issuer, status, eff_from, eff_to))

    def add_edge(self, a: str, b: str, family: str, raw: str = "") -> None:
        a, b = sorted((a, b))
        if a != b:
            self.db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?)", (a, b, family, raw))

    def remove_doc(self, id: str) -> None:
        self.db.execute("DELETE FROM edges WHERE a=? OR b=?", (id, id))
        self.db.execute("DELETE FROM docs WHERE id=?", (id,))

    def commit(self) -> None:
        self.db.commit()

    # ---- queries ----
    COLS = ("id", "key", "title", "issuer", "status", "eff_from", "eff_to")

    def doc(self, id: str) -> dict | None:
        r = self.db.execute(f"SELECT {','.join(self.COLS)} FROM docs WHERE id=?", (id,)).fetchone()
        return dict(zip(self.COLS, r)) if r else None

    def ids(self, key: str) -> list[str]:
        if not key:   # documents without a law number do not share an identity
            return []
        return [r[0] for r in self.db.execute("SELECT id FROM docs WHERE key=?", (key,))]

    def valid_on(self, id: str, as_of: str | date) -> bool | None:
        """True / False from the dates, None when the document or its dates are unknown. as_of 'yyyy-mm-dd'."""
        d = self.doc(id)
        if not d or not (d["eff_from"] or d["eff_to"]):
            return None
        t = str(as_of)
        return (not d["eff_from"] or d["eff_from"] <= t) and (not d["eff_to"] or t < d["eff_to"])

    def neighbors(self, key: str, families: tuple[str, ...] = (), as_of: str | date | None = None, limit: int = 20) -> list[dict]:
        """Documents linked to any document carrying law number `key`. as_of drops neighbours known NOT to be in force that day."""
        out = []
        for me in self.ids(key):
            q = "SELECT CASE WHEN a=? THEN b ELSE a END AS other, family, GROUP_CONCAT(raw, ' | ') FROM edges WHERE (a=? OR b=?)"
            args: list = [me, me, me]
            if families:
                q += f" AND family IN ({','.join('?' * len(families))})"
                args += list(families)
            q += " GROUP BY other, family"   # vbpl lists a relation on both documents: one row per (neighbour, family)
            for other, fam, raw in self.db.execute(q, args).fetchall():
                if as_of is not None and self.valid_on(other, as_of) is False:
                    continue
                d = self.doc(other) or {}
                out.append({"id": other, "key": d.get("key", ""), "family": fam, "raw": raw, "title": d.get("title", ""), "status": d.get("status", "")})
        return out[:limit]

    def related(self, key: str, hops: int = 1, families: tuple[str, ...] = ("amends", "replaces", "repeals", "expires"),
                per_hop: int = 5, max_total: int = 12) -> list[dict] | None:
        """Breadth-first by hop: [{"hop": 1, "total": n, "docs": [...]}, ...]; a document is listed once, at its first hop.
        `total` is the true number of documents at that hop; `docs` holds at most `per_hop` of them and `max_total` over all hops."""
        seen = set(self.ids(key))
        if not seen:
            return None   # unknown law (an empty list means: known, no such relation)
        frontier, out, shown = set(seen), [], 0
        for hop in range(1, max(1, hops) + 1):
            nxt: dict[str, dict] = {}
            for i in frontier:
                q = "SELECT CASE WHEN a=? THEN b ELSE a END, family FROM edges WHERE (a=? OR b=?)"
                args: list = [i, i, i]
                if families:
                    q += f" AND family IN ({','.join('?' * len(families))})"
                    args += list(families)
                for other, fam in self.db.execute(q, args):
                    if other not in seen and other not in nxt:
                        d = self.doc(other) or {}
                        nxt[other] = {"id": other, "key": d.get("key", ""), "family": fam, "title": d.get("title", ""), "status": d.get("status", "")}
            if not nxt:
                break
            seen |= set(nxt)
            docs = list(nxt.values())[: min(per_hop, max(0, max_total - shown))]
            shown += len(docs)
            out.append({"hop": hop, "total": len(nxt), "docs": docs})
            frontier = set(nxt)
        return out

    def amendments(self, key: str, limit: int = 20) -> list[dict]:
        return self.neighbors(key, ("amends", "replaces", "repeals", "expires"), limit=limit)

    def counts(self) -> dict:
        return {"docs": self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0], "edges": self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]}

    # ---- bulk build ----
    @classmethod
    def build(cls, raw: Path = Path("data/raw/vbpl"), path: str | Path = "data/kg/kg.sqlite") -> "KG":
        import polars as pl

        from corpus.keys import law_key

        meta, rel = pl.read_parquet(raw / "metadata.parquet"), pl.read_parquet(raw / "relationships.parquet")
        kg = cls(path)
        known = set()
        for r in meta.iter_rows(named=True):
            kg.upsert_doc(r["id"], law_key(r["so_ky_hieu"] or ""), r["title"] or "", r["co_quan_ban_hanh"] or "", r["tinh_trang_hieu_luc"] or "",
                          iso(r["ngay_co_hieu_luc"]), iso(r["ngay_het_hieu_luc"]))
            known.add(r["id"])
        for a, b, raw_name in rel.iter_rows():
            if a in known and b in known:
                kg.add_edge(a, b, FAMILY.get(raw_name, "other"), raw_name)
        kg.commit()
        return kg
