"""Alternative ways to cut one long article into pieces (chunking ablation). Pure functions, no index or model needed
except `semantic_split`, which takes the embedding function as an argument."""
import re

SENT_END = re.compile(r"(?<=[.;?!])\s+")
MARKER = re.compile(r"^(?:\d{1,2}|[a-zđ])[.)]$")  # list markers "1." "a)" must stay with the text they introduce


def sentences(body: str) -> list[str]:
    """Paragraph lines, then sentence ends; bare list markers are glued to the following text."""
    out: list[str] = []
    for line in body.split("\n"):
        pending = ""
        for frag in SENT_END.split(line.strip()):
            if not frag:
                continue
            frag = f"{pending} {frag}".strip() if pending else frag
            pending = frag if MARKER.match(frag) else ""
            if not pending:
                out.append(frag)
        if pending:
            out.append(pending)
    return out


def pack_sentences(body: str, max_chars: int) -> list[str]:
    """Greedy: add sentences until the next one would exceed max_chars; a sentence is never cut."""
    pieces: list[str] = []
    cur = ""
    for s in sentences(body):
        if cur and len(cur) + 1 + len(s) > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces


CLAUSE = re.compile(r"^\d{1,3}[.)]\s")


def khoan_units(body: str) -> list[str]:
    """Clause units: a numbered line starts a unit, lettered points stay with it; the heading joins the first clause."""
    units: list[str] = []
    lead = ""  # text before the first numbered clause (the article heading)
    for line in (ln.strip() for ln in body.split("\n")):
        if not line:
            continue
        if CLAUSE.match(line):
            units.append(f"{lead}\n{line}" if lead and not units else line)
        elif units:
            units[-1] += "\n" + line
        else:
            lead = f"{lead}\n{line}" if lead else line
    return units or ([lead] if lead else [])


def pack_khoan(body: str, max_chars: int) -> list[str]:
    """Merge consecutive clauses up to max_chars; a clause that alone exceeds it is cut at sentence ends."""
    pieces: list[str] = []
    cur = ""
    for u in khoan_units(body):
        if len(u) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(pack_sentences(u, max_chars))
        elif cur and len(cur) + 1 + len(u) > max_chars:
            pieces.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        pieces.append(cur)
    return pieces


def semantic_split(body: str, embed, pct: float, max_chars: int) -> list[str]:
    """Break where the cosine distance between neighbouring sentences exceeds the `pct` percentile of all
    neighbour distances in this article; each group is then packed to max_chars (a group is never merged with its
    neighbour, so the topic boundary stays). `embed(list[str]) -> list[vector]`."""
    import numpy as np

    sents = sentences(body)
    if len(sents) < 2:
        return pack_sentences(body, max_chars)
    v = np.asarray(embed(sents), dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    dist = 1.0 - (v[:-1] * v[1:]).sum(axis=1)
    cut = np.percentile(dist, pct)
    groups, cur = [], [sents[0]]
    for s, d in zip(sents[1:], dist, strict=True):
        if d > cut:
            groups.append(cur)
            cur = [s]
        else:
            cur.append(s)
    groups.append(cur)
    return [p for g in groups for p in pack_sentences("\n".join(g), max_chars)]
