"""Gold (document, Điều) pairs per question, parsed from the reference answers — the target for retrieval eval.

    GoldSet().build()  -> data/processed/gold.parquet: qid, docs (law keys), pairs ["<law key>#<Điều>"], primary
    GoldSet().load()   -> {qid: Gold}

An answer like "Căn cứ khoản 1 Điều 5 Thông tư 12/2021/TT-BGTVT …" gives the pair (12/2021/TT-BGTVT, 5). Each
"Điều N" goes to the nearest document cited after it (within WINDOW chars), else to the question's law_id.
"Điều này" / "Điều đó" are skipped. Laws cited by name ("Luật Đất đai 2024") are resolved by vbpl title + year.
Answers are the labels of train / dev / test alike; this file is for scoring only, never for training.
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from corpus.keys import law_key
from corpus.sources.citations import Citations

OUT = Path("data/processed/gold.parquet")
ARTICLE = re.compile(r"\bĐiều\s+(\d+[a-zđ]?)\b(?!\s*(?:này|đó|trên|dưới))")
WINDOW = 200


@dataclass(frozen=True)
class Gold:
    qid: int
    docs: frozenset[str]     # law keys of every cited document (+ law_id)
    pairs: frozenset[str]    # "<law key>#<Điều>"
    primary: str             # law_id key, else the first cited document, else ""


class GoldSet:
    def __init__(self, qa: Path = Path("data/processed/qa.parquet"), out: Path = OUT):
        self.qa, self.out = Path(qa), Path(out)
        self._cit = None

    def parse(self, answer: str, law_id: str = "") -> tuple[set[str], set[str]]:
        """(docs, pairs) of one answer."""
        text = unicodedata.normalize("NFC", answer or "")
        self._cit = self._cit or Citations(self.qa)
        refs = [(r, self._cit.resolve(r)) for r in Citations.spans(text)]
        refs = [(r, law_key(n)) for r, n in refs if n]
        fallback = law_key(law_id) if re.search(r"\d", law_id or "") else ""  # not "Không trích xuất được"
        docs = {k for _, k in refs} | ({fallback} if fallback else set())
        pairs = set()
        for m in ARTICLE.finditer(text):
            after = [k for r, k in refs if 0 <= r.start - m.end() <= WINDOW]
            doc = after[0] if after else fallback
            if doc:
                pairs.add(f"{doc}#{m.group(1)}")
        return docs, pairs

    def build(self) -> Path:
        import polars as pl

        rows = []
        for r in pl.read_parquet(self.qa).iter_rows(named=True):
            law_id = r["law_id"] if re.search(r"\d", r["law_id"] or "") else ""
            docs, pairs = self.parse(r["answer"], law_id)
            rows.append({"qid": r["qid"], "docs": sorted(docs), "pairs": sorted(pairs),
                         "primary": law_key(law_id) if law_id else next(iter(sorted(docs)), "")})
        df = pl.DataFrame(rows, schema={"qid": pl.Int64, "docs": pl.List(pl.String), "pairs": pl.List(pl.String),
                                        "primary": pl.String})
        self.out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self.out)
        print(f"gold: {len(df)} questions, {(df['docs'].list.len() > 0).sum()} with a document, "
              f"{(df['pairs'].list.len() > 0).sum()} with a Điều -> {self.out}")
        return self.out

    def load(self) -> dict[int, Gold]:
        import polars as pl

        if not self.out.exists():
            self.build()
        return {r["qid"]: Gold(r["qid"], frozenset(r["docs"]), frozenset(r["pairs"]), r["primary"])
                for r in pl.read_parquet(self.out).iter_rows(named=True)}
