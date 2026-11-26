"""Clause hint: cut the retrieved passages into khoản units and name the ones whose words overlap the question most, as one extra line in the prompt.
Lexical only (idf-weighted overlap of normalised syllables, idf over the units of the current context), no model call.

    clause_hint(question, passages, top=2) -> "Khoản có thể liên quan nhất: khoản 1 Điều 17; khoản 3 Điều 17." (or "" when nothing overlaps)
"""
import math
import re
from dataclasses import dataclass

from evaluate.metrics import normalize_text
from records import Passage

HEADING = re.compile(r"^\s*Điều\s+(\d+[a-zđ]?)\s*[.:]")
KHOAN = re.compile(r"^\s*(\d+)\s*[.\)]\s")
STOP = frozenset("là của và các có được trong cho một những này theo khi nào gì như thế nào đối với không bao nhiêu thì bị về từ đến".split())


@dataclass(frozen=True)
class Unit:
    doc: str
    dieu: str
    khoan: str | None
    text: str


def split_units(p: Passage) -> list[Unit]:
    """One unit per numbered khoản (with its a)/b) items) of each article in the passage; an article without numbered khoản is one unit with khoan None."""
    units, dieu, khoan, buf = [], None, None, []

    def flush():
        if dieu is not None and buf:
            units.append(Unit(p.doc, dieu, khoan, "\n".join(buf)))

    for line in p.text.split("\n"):
        h = HEADING.match(line)
        k = KHOAN.match(line)
        if h:
            flush()
            dieu, khoan, buf = h.group(1), None, []
        elif k and dieu is not None:
            flush()
            khoan, buf = k.group(1), [line]
        elif dieu is not None:
            buf.append(line)
    flush()
    return units


def words(text: str) -> list[str]:
    return [w for w in normalize_text(text).split() if w not in STOP and len(w) > 1]


def clause_hint(question: str, passages: list[Passage], top: int = 2) -> str:
    units = [u for p in passages for u in split_units(p)]
    if not units:
        return ""
    docs = [set(words(u.text)) for u in units]
    df: dict[str, int] = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    q = set(words(question))
    scored = []
    for u, d in zip(units, docs):
        s = sum(math.log(1 + len(units) / df[w]) for w in q & d)
        if s > 0:
            scored.append((s, u))
    if not scored:
        return ""
    scored.sort(key=lambda x: -x[0])
    seen, names = set(), []
    for _, u in scored:
        name = f"khoản {u.khoan} Điều {u.dieu}" if u.khoan else f"Điều {u.dieu}"
        if name not in seen:
            seen.add(name)
            names.append(name)
        if len(names) == top:
            break
    return "Khoản có thể liên quan nhất: " + "; ".join(names) + "."
