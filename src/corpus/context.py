"""Rule-based "where does this Điều sit" context (no LLM): Chương / Mục, scope sentence, one compact context line."""
import re

_CHUONG = re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)\b\s*[.:\-–]?\s*(.*)$", re.I)
_MUC = re.compile(r"^\s*Mục\s+(\d+|[IVXLCDM]+)\b\s*[.:\-–]?\s*(.*)$", re.I)
_DIEU = re.compile(r"^\s*Điều\s+(\d+[a-zA-Z]?)\s*\.")


def _nice(title: str) -> str:
    title = title.strip()
    return title[:1].upper() + title[1:].lower() if title.isupper() else title


def _heading(label: str, num: str, title: str) -> str:
    title = _nice(title)
    return f"{label} {num}: {title}" if title else f"{label} {num}"


def chapter_of(doc_text: str, dieu: str) -> str:
    """Chương (+ " / Mục ...") governing "Điều <dieu>"; "" without chapters or when the Điều is absent."""
    lines = doc_text.splitlines()
    chapter = muc = ""
    for i, line in enumerate(lines):
        m = _DIEU.match(line)
        if m and m.group(1).lower() == str(dieu).lower():
            return chapter + (f" / {muc}" if chapter and muc else muc)
        c = _CHUONG.match(line)
        if c:
            title = c.group(2)
            if not title:
                nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
                if nxt and not _DIEU.match(nxt) and not _MUC.match(nxt):
                    title = nxt
            chapter, muc = _heading("Chương", c.group(1).upper(), title), ""
            continue
        m = _MUC.match(line)
        if m:
            muc = _heading("Mục", m.group(1), m.group(2))
    return ""


def scope_of(doc_text: str, max_chars: int = 200) -> str:
    """First sentence of the Điều 1 body, cut at a word boundary to max_chars; "" without Điều 1."""
    lines = doc_text.splitlines()
    for i, line in enumerate(lines):
        m = _DIEU.match(line)
        if not (m and m.group(1) == "1"):
            continue
        body = []
        for l in lines[i + 1:]:
            if _DIEU.match(l) or _CHUONG.match(l) or _MUC.match(l):
                break
            if l.strip():
                body.append(l.strip())
        text = re.sub(r"^\d+\.\s+", "", " ".join(body))
        first = re.split(r"(?<=[.;:!?])\s", text, maxsplit=1)[0]
        if len(first) > max_chars:
            first = first[:max_chars + 1].rsplit(" ", 1)[0] if " " in first[:max_chars + 1] else first[:max_chars]
            first = first[:max_chars].rstrip()
        return first
    return ""


def context_line(title: str, chapter: str, scope: str) -> str:
    """"Văn bản: <title> | Phần: <chapter> | Phạm vi: <scope>" without the empty parts, at most 400 chars."""
    parts = [f"{k}: {v}" for k, v in (("Văn bản", title), ("Phần", chapter), ("Phạm vi", scope)) if v]
    line = " | ".join(parts)
    return line if len(line) <= 400 else line[:400].rstrip()


def enrich(text: str, doc_text: str, dieu: str, title: str) -> str:
    """Chunk text with the context line as first line; unchanged if it already starts with it."""
    line = context_line(title, chapter_of(doc_text, dieu), scope_of(doc_text))
    return text if not line or text.startswith(line) else f"{line}\n{text}"


def pool_with_context(pool_passages, doc_text_of):
    """Copy of an embed-bench pool with enriched texts. pid / doc / score are kept; passages whose document text is unknown
    (`doc_text_of(doc)` -> None) or that have no "Điều N." heading are returned unchanged."""
    from dataclasses import replace

    out = []
    for p in pool_passages:
        doc_text = doc_text_of(p.doc)
        lines = p.text.splitlines()
        idx = next((i for i, l in enumerate(lines) if _DIEU.match(l)), None)
        if doc_text is None or idx is None:
            out.append(p)
            continue
        dieu = _DIEU.match(lines[idx]).group(1)
        title = lines[0].strip() if idx > 0 else ""
        out.append(replace(p, text=enrich(p.text, doc_text, dieu, title)))
    return out
