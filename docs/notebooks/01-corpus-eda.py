"""Corpus exploration (report section "Data exploration").

Run:  uv run --with matplotlib python docs/notebooks/01-corpus-eda.py
Out:  docs/report/corpus-eda.md, docs/report/figures/corpus-*.png

Reads data/processed/corpus/*.txt and the QA law ids of train + dev only (test stays unseen).
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/processed/corpus"
OUT = ROOT / "docs/report"
FIG = OUT / "figures"

DOC_TYPES = ["VĂN BẢN HỢP NHẤT", "THÔNG TƯ LIÊN TỊCH", "THÔNG TƯ LIÊN BỘ", "NGHỊ QUYẾT LIÊN TỊCH", "LUẬT", "BỘ LUẬT",
             "PHÁP LỆNH", "NGHỊ ĐỊNH", "NGHỊ QUYẾT", "THÔNG TƯ", "QUYẾT ĐỊNH", "CHỈ THỊ", "KẾ HOẠCH", "THÔNG BÁO",
             "HƯỚNG DẪN", "CÔNG VĂN", "BÁO CÁO", "LỆNH", "ĐIỀU LỆ", "QUY ĐỊNH", "CHƯƠNG TRÌNH", "ĐỀ ÁN"]
REL_HEADERS = {  # trailing metadata blocks, ASCII as written in the files
    "Van ban duoc can cu": "based_on", "Van ban duoc dan chieu": "cites", "Van ban huong dan": "guided_by",
    "Van ban duoc huong dan": "guides", "Van ban sua doi bo sung": "amended_by", "Van ban bi sua doi bo sung": "amends",
    "Van ban thay the": "replaced_by", "Van ban bi thay the": "replaces", "Van ban lien quan cung noi dung": "related",
    "Van ban lien quan ngon ngu": "translation",
}
REL_RE = re.compile(r"^(" + "|".join(REL_HEADERS) + r"):\s*$", re.M)
DIEU_RE = re.compile(r"(?:^|\n)\s*Điều\s+\d+\s*[.:]", re.M)
CHUONG_RE = re.compile(r"(?:^|\n)\s*Chương\s+[IVXLC\d]+", re.M)
NUMBER_RE = re.compile(r"Số\s*:\s*([0-9A-Za-zĐđ/_.\-–]+)")
YEAR_RE = re.compile(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+((?:19|20)\d{2})")


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_id(s: str) -> str:
    return re.sub(r"[^0-9a-zđ]", "", s.lower())


def issuer_level(head: str) -> str:
    h = head.upper()
    if "QUỐC HỘI" in h:
        return "Quốc hội"
    if re.search(r"ỦY BAN NHÂN DÂN|UỶ BAN NHÂN DÂN|UBND|HỘI ĐỒNG NHÂN DÂN|HĐND|\bSỞ\b|TỈNH|THÀNH PHỐ|HUYỆN|QUẬN", h):
        return "Địa phương"
    if re.search(r"CHÍNH PHỦ|THỦ TƯỚNG", h):
        return "Chính phủ"
    if re.search(r"\bBỘ\b|NGÂN HÀNG NHÀ NƯỚC|TÒA ÁN|TOÀ ÁN|VIỆN KIỂM SÁT|KIỂM TOÁN|BẢO HIỂM XÃ HỘI|TỔNG CỤC|ỦY BAN", h):
        return "Bộ, ngành"
    return "Khác"


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = REL_RE.search(text)
    body, tail = (text[: m.start()], text[m.start():]) if m else (text, "")
    head = flat(body[:1500])
    issuer = flat(re.split(r"CỘNG HÒA|CỘNG HOÀ|VIỆT\s+NAM", body[:600], maxsplit=1)[0])
    hits = [(m.start(), -len(t), t) for t in DOC_TYPES for m in [re.search(rf"(?:^|\s){t}(?:\s|$)", head)] if m]
    dtype = min(hits)[2] if hits else "Khác"  # earliest type word in the header; longest wins a tie
    num = NUMBER_RE.search(head)
    year = YEAR_RE.search(head)
    rels = Counter()
    for block in REL_RE.finditer(tail):
        end = tail.find("\n\n", block.end())
        seg = tail[block.end(): end if end > 0 else None]
        rels[REL_HEADERS[block.group(1)]] += seg.count("(ID:")
    noi_nhan = body.find("Nơi nhận")
    return {
        "file": path.name,
        "chars": len(body),
        "words": len(body.split()),
        "type": dtype,
        "level": issuer_level(issuer),
        "number": num.group(1) if num else "",
        "year": int(year.group(1)) if year else None,
        "dieu": len(DIEU_RE.findall(body)),
        "chuong": len(CHUONG_RE.findall(body)),
        "noi_nhan_tail": len(body) - noi_nhan if noi_nhan > 0 else 0,
        "rel_total": sum(rels.values()),
        **{f"rel_{k}": rels.get(k, 0) for k in REL_HEADERS.values()},
        "hash": hashlib.md5(flat(body).lower().encode()).hexdigest(),
        "rel_text": tail,
    }


def md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(f"{v:,}" if isinstance(v, int) else (f"{v:.1f}" if isinstance(v, float) else str(v))
                               for v in r) + " |" for r in df.iter_rows()]
    return "\n".join(rows)


def bar(series: pl.DataFrame, x: str, y: str, title: str, name: str, horizontal: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    (ax.barh if horizontal else ax.bar)(series[x].cast(str).to_list(), series[y].to_list(), color="#4C72B0")
    if horizontal:
        ax.invert_yaxis()
    else:
        ax.tick_params(axis="x", labelrotation=60)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    return f"![{title}](figures/{name})"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    docs = pl.DataFrame([parse(p) for p in sorted(CORPUS.glob("*.txt"))])
    n = len(docs)
    out = ["# Corpus exploration", "",
           f"Source: `Dataset.zip` → `data/processed/corpus/`. QA law ids from **train + dev only** (test unseen). "
           f"Script: `docs/notebooks/01-corpus-eda.py`.", ""]

    # 1. size
    q = lambda c, p: int(docs[c].quantile(p))
    est_chunks = int((docs["chars"] / 1950).ceil().sum())
    out += ["## 1. Size", "",
            md_table(pl.DataFrame({
                "metric": ["documents", "total words", "words / doc p10", "median", "p90", "max",
                           "near-empty docs (< 50 words)", "est. 2000-char chunks"],
                "value": [n, int(docs["words"].sum()), q("words", .1), q("words", .5), q("words", .9),
                          int(docs["words"].max()), int((docs["words"] < 50).sum()), est_chunks]})), ""]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(docs["words"].clip(upper_bound=20000).to_list(), bins=60, color="#4C72B0")
    ax.set_yscale("log"); ax.set_xlabel("words per document (clipped at 20k)"); ax.set_title("Document length")
    fig.tight_layout(); fig.savefig(FIG / "corpus-length.png", dpi=130); plt.close(fig)
    out += ["![Document length](figures/corpus-length.png)", ""]

    # 2. who issues what
    lv = docs.group_by("level").len().sort("len", descending=True).with_columns((pl.col("len") / n * 100).round(1).alias("%"))
    ty = docs.group_by("type").len().sort("len", descending=True).with_columns((pl.col("len") / n * 100).round(1).alias("%"))
    out += ["## 2. Issuer level and document type", "", md_table(lv), "",
            bar(lv, "level", "len", "Issuer level", "corpus-level.png"), "", md_table(ty.head(15)), "",
            bar(ty.head(12), "type", "len", "Document type (from header)", "corpus-type.png"), ""]

    # 3. time
    yr = docs.drop_nulls("year")
    dec = yr.with_columns((pl.col("year") // 5 * 5).alias("from")).group_by("from").len().sort("from")
    out += ["## 3. Issue year", "",
            f"Year found in header for {len(yr):,}/{n:,} docs; range {yr['year'].min()}–{yr['year'].max()}, "
            f"median {int(yr['year'].median())}; 2024+: {int((yr['year'] >= 2024).sum()):,}.", "",
            bar(dec, "from", "len", "Documents per 5-year bin", "corpus-year.png", horizontal=False), ""]

    # 4. structure + noise
    has_dieu = int((docs["dieu"] > 0).sum())
    out += ["## 4. Structure and noise", "",
            md_table(pl.DataFrame({
                "metric": ["docs with ≥1 Điều", "docs with Chương", "Điều total", "Điều / doc (median, docs with Điều)",
                           "docs with 'Nơi nhận' block", "chars after 'Nơi nhận' (median)",
                           "exact duplicate bodies (extra copies)", "file names ending _1.._9 (variants)"],
                "value": [has_dieu, int((docs["chuong"] > 0).sum()), int(docs["dieu"].sum()),
                          int(docs.filter(pl.col("dieu") > 0)["dieu"].median()), int((docs["noi_nhan_tail"] > 0).sum()),
                          int(docs.filter(pl.col("noi_nhan_tail") > 0)["noi_nhan_tail"].median()),
                          n - docs["hash"].n_unique(), int(docs["file"].str.contains(r"_\d\.txt$").sum())]})), "",
            "Text is copied from HTML: line breaks fall mid-sentence (`Số:\\n 002-TT-LB`), so chunkers should "
            "flatten whitespace first; the header (quốc hiệu, số, ngày) and the `Nơi nhận` tail are boilerplate.", ""]

    # 5. relations graph
    rel_cols = [c for c in docs.columns if c.startswith("rel_") and c not in ("rel_total", "rel_text")]
    rel = pl.DataFrame({"relation": [c[4:] for c in rel_cols],
                        "docs having it": [int((docs[c] > 0).sum()) for c in rel_cols],
                        "edges": [int(docs[c].sum()) for c in rel_cols]}).sort("edges", descending=True)
    out += ["## 5. Built-in relation metadata (a ready-made document graph)", "",
            f"Every file ends with blocks such as `Van ban duoc can cu:` listing related documents with an ID. "
            f"{int((docs['rel_total'] > 0).sum()):,} docs have at least one block; {int(docs['rel_total'].sum()):,} edges in total.", "",
            md_table(rel), "",
            "Use: expand retrieval to guiding / amending documents, prefer the newest version (`replaced_by`), "
            "no LLM extraction needed (lightweight graph, roadmap phase 4).", ""]

    # 6. coverage of QA law ids (train + dev)
    split = json.loads((ROOT / "data/processed/split.json").read_text())
    keep = set(split["train"]) | set(split["dev"])
    qa = pl.read_parquet(ROOT / "data/processed/qa.parquet").filter(pl.col("qid").is_in(list(keep)))
    have = {norm_id(x) for x in docs["number"].to_list() if x} | {norm_id(f.removesuffix(".txt")) for f in docs["file"]}
    rel_blob = norm_id(" ".join(docs["rel_text"].to_list()))
    laws = qa["law_id"].fill_null("").to_list()
    ids = [norm_id(x) for x in laws if x]
    in_corpus = sum(i in have for i in ids)
    mentioned = sum(i in rel_blob for i in ids if i not in have)
    years = Counter(int(m.group(0)) for x in laws for m in [re.search(r"(?:19|20)\d{2}", x)] if m)
    out += ["## 6. Does the corpus hold the laws the questions cite? (train + dev)", "",
            md_table(pl.DataFrame({
                "metric": ["QA pairs (train + dev)", "with a cited law id", "cited law found in corpus",
                           "not in corpus but named in a relation block", "distinct cited laws"],
                "value": [len(qa), len(ids), in_corpus, mentioned, len(set(ids))]})), "",
            "Cited-law year (QA): " + ", ".join(f"{y}: {c}" for y, c in sorted(years.items()) if 2015 <= y <= 2026), "",
            "The questions ask about 2024–2025 law that the corpus mostly lacks; this is why `law_hit` is ~0 for B0/A1 "
            "and why the vbpl extra corpus is added.", ""]

    (OUT / "corpus-eda.md").write_text("\n".join(out), encoding="utf-8")
    docs.drop("rel_text").write_parquet(ROOT / "data/processed/corpus_stats.parquet")
    print("\n".join(out))


if __name__ == "__main__":
    main()
