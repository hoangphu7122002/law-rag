"""Question profile for the ReAct agent (docs/research/teacher-data-plan.md, Stage 1c-2): layer 1 = rules, no model.

    question_profile(q) -> {"laws": [law_key], "partial": ["39/2016"], "law_names": [(name, year)], "dieu": ["51"],
                            "years": [2025], "type": "thoi_han" | ..., "flags": ["du_thao", ...]}
    needs_fallback(profile) -> True when nothing specific was found (the small-LLM fallback then fills the profile; its output is
    validated against the store, never trusted).
Patterns are data (tables below); the number / named-law patterns are shared with the citation parser.
"""
import re

from corpus.keys import law_key
from corpus.sources.citations import BARE_NUMBER, DOC_TYPES, LAW, NUMBER

PARTIAL = re.compile(rf"{DOC_TYPES}\s+(?:số\s+)?(\d+[a-zA-Z]?/(?:19|20)\d{{2}})(?![\w/-])")
DIEU = re.compile(r"[Đđ]iều\s+(\d+[a-zđ]?(?:\s*(?:,|và)\s*\d+[a-zđ]?)*)")
YEAR = re.compile(r"\b(?:năm\s+)?((?:19|20)\d{2})\b")
LAW_NAME = re.compile(r"\b(Bộ luật|Luật)\s+([A-ZĐÂ][^\d.;:()\[\]\n]{2,90}?)\s+(?:năm\s+)?((?:19|20)\d{2})\b")

TYPES = [   # first match wins; keyword tables, not code
    ("muc_phat", r"mức phạt|xử phạt|phạt tiền|bị phạt"),
    ("thoi_han", r"thời hạn|bao lâu|trong vòng|kể từ ngày|thời gian"),
    ("tham_quyen", r"thẩm quyền|cơ quan nào|ai (?:có|được)|do ai"),
    ("truong_hop", r"trường hợp nào|những trường hợp|trong trường hợp"),
    ("dieu_kien", r"điều kiện|tiêu chuẩn|yêu cầu gì"),
    ("thu_tuc", r"thủ tục|hồ sơ|trình tự|quy trình"),
    ("quyen_nghia_vu", r"quyền|nghĩa vụ|trách nhiệm|nhiệm vụ"),
    ("muc_so", r"bao nhiêu|mức|tỷ lệ|hạn mức"),
]
FLAGS = [
    ("du_thao", r"dự thảo"), ("thong_bao", r"thông báo"), ("phu_luc", r"phụ lục"),
    ("mau_bieu", r"mẫu|biểu mẫu"), ("hien_hanh", r"hiện hành|mới nhất|hiện nay"),
]


def _dedup(xs):
    return list(dict.fromkeys(xs))


def question_profile(question: str) -> dict:
    q = question or ""
    low = q.lower()
    laws = _dedup(law_key(m.group(1)) for pat in (NUMBER, BARE_NUMBER) for m in pat.finditer(q))
    short = [m.group(1) for m in PARTIAL.finditer(q)]
    partial = _dedup(s for s in short if not any(l.startswith(law_key(s) + "/") for l in laws))
    dieu = _dedup(n.strip() for m in DIEU.finditer(q) for n in re.split(r"\s*(?:,|và)\s*", m.group(1)))
    names = _dedup((f"{m.group(1)} {m.group(2).strip()}", int(m.group(3))) for m in LAW_NAME.finditer(q))
    named_years = {y for _, y in names}
    number_years = {int(y) for s in [*laws, *short] for y in re.findall(r"/((?:19|20)\d{2})(?:/|$)", s)}
    years = _dedup(int(m.group(1)) for m in YEAR.finditer(q)
                   if int(m.group(1)) not in named_years | number_years and "/" not in q[max(0, m.start() - 1):m.start()])
    kind = next((name for name, pat in TYPES if re.search(pat, low)), None)
    return {"laws": laws, "partial": partial, "law_names": names, "dieu": dieu, "years": years,
            "type": kind, "flags": [name for name, pat in FLAGS if re.search(pat, low)]}


def needs_fallback(profile: dict) -> bool:
    """Nothing specific: no law number / name, no Điều, no flag."""
    return not (profile["laws"] or profile["partial"] or profile["law_names"] or profile["dieu"] or profile["flags"])
