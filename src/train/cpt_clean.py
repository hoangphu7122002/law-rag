"""Cleaning and curation of external CPT text (docs/research/cpt: quality + dedup + PII + no eval leakage matter more than raw volume).

Pure helpers used by scripts/cpt_build_v2.py: strip_html, normalize, quality (reason or None), mask_pii (court judgments), Dedup (exact document, repeated
long paragraphs, repeated document numbers), chunk_doc (paragraph chunks with the document title as first line, like the retrieval prompts).
"""
import hashlib
import html
import re
import unicodedata

VI_DIACRITICS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
MOJIBAKE = re.compile(r"Ã.|Ä.|áº|á»|Æ°|�")
BLOCK = re.compile(r"</?(?:p|div|br|tr|li|ul|ol|h[1-6]|table|section|article|blockquote|pre)\b[^>]*>", re.I)
CELL = re.compile(r"</?(?:td|th)\b[^>]*>", re.I)
DROP = re.compile(r"<(script|style|head|noscript)\b.*?</\1>", re.I | re.S)
TAG = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    s = DROP.sub(" ", s)
    s = BLOCK.sub("\n", s)
    s = CELL.sub(" \t ", s)
    return normalize(html.unescape(TAG.sub("", s)))


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s.replace(" ", " ").replace("\r", ""))
    s = "".join(c for c in s if c in "\n\t" or unicodedata.category(c) not in ("Cc", "Cf", "Cs"))
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in s.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def quality(t: str, min_chars: int = 300) -> str | None:
    """Reason to drop a document, or None."""
    if len(t) < min_chars:
        return "too_short"
    if len(MOJIBAKE.findall(t)) > 5 + len(t) // 2000:
        return "mojibake"
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 0.4 * len(t):
        return "symbols"
    low = t.lower()
    if sum(c in VI_DIACRITICS for c in low) < 0.02 * len(letters):
        return "not_vietnamese"
    lines = [l for l in t.split("\n") if len(l) > 12]
    if lines and len(set(lines)) < 0.4 * len(lines) and len(lines) > 8:
        return "repetitive"
    return None


PHONE = re.compile(r"(?<!\d)(?:\+84|0)\d{9,10}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IDNUM = re.compile(r"(?<!\d)\d{12}(?!\d)|(?<!\d)\d{9}(?!\d)")
CAP = r"[A-ZÀ-Ỹ][a-zà-ỹ]+"
NAME = re.compile(r"\b((?:[Bb]ị cáo|[Bb]ị hại|[Nn]guyên đơn|[Bb]ị đơn|[Nn]gười có quyền lợi, nghĩa vụ liên quan|[Oo]ng|[Bb]à|[Aa]nh|[Cc]hị|[Ee]m|[Cc]háu|[Cc]ụ|ông|bà)\s+)" + rf"({CAP}(?:\s+{CAP}){{1,3}})")


def mask_pii(t: str) -> str:
    t = EMAIL.sub("[EMAIL]", t)
    t = PHONE.sub("[SĐT]", t)
    t = IDNUM.sub("[SỐ]", t)
    return NAME.sub(lambda m: m.group(1) + "[TÊN]", t)


def _h(s: str) -> str:
    return hashlib.blake2b(re.sub(r"\W+", " ", s.lower()).strip().encode(), digest_size=12).hexdigest()


class Dedup:
    """accept(doc_id, text, key=None) -> text with already-seen long paragraphs removed, or None if the document is a duplicate / has nothing new."""

    def __init__(self, min_par: int = 150, min_new: int = 60):
        self.docs, self.pars, self.keys, self.min_par, self.min_new = set(), set(), set(), min_par, min_new

    def accept(self, doc_id: str, text: str, key: str | None = None) -> str | None:
        if key:
            k = re.sub(r"\s+", "", key).lower()
            if k in self.keys:
                return None
        hd = _h(text)
        if hd in self.docs:
            return None
        out, new = [], 0
        for p in text.split("\n"):
            if len(p) >= self.min_par:
                hp = _h(p)
                if hp in self.pars:
                    continue
                self.pars.add(hp)
            new += len(p)
            out.append(p)
        res = "\n".join(out).strip()
        if new < self.min_new:
            return None
        self.docs.add(hd)
        if key:
            self.keys.add(re.sub(r"\s+", "", key).lower())
        return res


def chunk_doc(text: str, title: str = "", max_chars: int = 3000) -> list[str]:
    """Paragraph-aligned chunks; every chunk starts with the document title line."""
    head = f"{title}\n" if title else ""
    out, buf = [], ""
    for p in text.split("\n"):
        while len(p) > max_chars:   # a very long paragraph is cut at a space
            cut = p.rfind(" ", 0, max_chars) or max_chars
            if buf:
                out.append(buf)
                buf = ""
            out.append(p[:cut]); p = p[cut:].lstrip()
        if buf and len(buf) + len(p) + 1 > max_chars:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf.strip():
        out.append(buf)
    return [head + c for c in out if c.strip()]
