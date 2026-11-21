"""Retrieval profile (the df.info() of a retrieval result) + sufficiency signals for the ReAct agent
(docs/research/teacher-data-plan.md, Stage 1c): code only, neutral, short, never the article text.

    text, signals = retrieval_profile(question_profile(q), passages, kg, toc=None, rng=None, shuffle=0.0)
signals (facts, not commands): law_missing / dieu_missing / flag_missing / same_family.
rng + shuffle: with probability `shuffle` the laws are listed in shuffled order (rank 1 is not privileged).
"""
import re

from corpus.keys import law_key, stem_key
from evaluate.retrieval_eval import HEADING

FLAG_PATTERNS = {"phu_luc": r"phụ lục", "mau_bieu": r"\bmẫu\b|biểu mẫu", "thong_bao": r"thông báo", "du_thao": r"dự thảo"}
MAX_LAWS, MAX_DIEU = 6, 6


def _titles(text: str) -> dict[str, str]:
    """{Dieu: heading} of the headings found in one passage."""
    return {m.group(1): re.sub(r"\s+", " ", text[m.end():].split("\n", 1)[0]).strip()[:60]
            for m in re.finditer(r"(?m)^\s*Điều\s+(\d+[a-zđ]?)\s*[.:]", text)}


def _kg_info(kg, key: str):
    if kg is None:
        return {}, []
    ids = kg.ids(key)
    doc = kg.doc(ids[0]) if ids else {}
    rel = kg.related(key, hops=1) or []
    return doc or {}, [d["key"] for h in rel for d in h["docs"] if d.get("key")]


def retrieval_profile(qp: dict, passages, kg=None, toc=None, rng=None, shuffle: float = 0.0) -> tuple[str, list[dict]]:
    laws: dict[str, dict] = {}   # law key -> {"dieu": {n: heading}, "title": str}, in rank order
    for p in passages:
        k = stem_key(p.doc)
        entry = laws.setdefault(k, {"dieu": {}, "title": (p.text.split("\n", 1)[0] or "")[:80], "text": ""})
        entry["dieu"].update(_titles(p.text))
        entry["text"] += " " + p.text[:400].lower()
    keys = list(laws)
    info = {k: _kg_info(kg, k) for k in keys}
    signals: list[dict] = []
    for want in qp["laws"]:
        if law_key(want) not in laws:
            signals.append({"kind": "law_missing", "law": want})
    for part in qp["partial"]:
        if not any(k.startswith(law_key(part) + "/") for k in keys):
            signals.append({"kind": "law_missing", "law": part})
    target = [law_key(w) for w in qp["laws"] if law_key(w) in laws] or ([keys[0]] if keys else [])
    for n in qp["dieu"]:
        for law in (qp["laws"] or target[:1]):
            if law_key(law) in laws and n not in laws[law_key(law)]["dieu"]:
                signals.append({"kind": "dieu_missing", "law": law_key(law), "dieu": n})
            elif law_key(law) not in laws and qp["laws"]:
                signals.append({"kind": "dieu_missing", "law": law_key(law), "dieu": n})
    blob = " ".join(e["text"] for e in laws.values())
    for flag in qp["flags"]:
        if flag in FLAG_PATTERNS and not re.search(FLAG_PATTERNS[flag], blob):
            signals.append({"kind": "flag_missing", "flag": flag})
    for k in keys:
        fam = sorted({k, *(x for x in info[k][1] if x in laws)})
        if len(fam) > 1 and not any(s["kind"] == "same_family" and s["laws"] == fam for s in signals):
            signals.append({"kind": "same_family", "laws": fam})
    order = list(keys)
    if rng is not None and shuffle > 0 and rng.random() < shuffle:
        rng.shuffle(order)
    lines = ["[HỒ SƠ TRUY XUẤT]"]
    asked = [f"số hiệu {', '.join(qp['laws'] + qp['partial'])}" if qp["laws"] or qp["partial"] else "", f"Điều {', '.join(qp['dieu'])}" if qp["dieu"] else "",
             f"dạng {qp['type']}" if qp["type"] else "", f"cờ {', '.join(qp['flags'])}" if qp["flags"] else ""]
    lines.append("Câu hỏi nêu: " + ("; ".join(a for a in asked if a) or "không nêu cụ thể"))
    for i, k in enumerate(order[:MAX_LAWS], 1):
        doc = info[k][0]
        bits = [doc.get("status") or "hiệu lực không rõ"] + ([f"hiệu lực từ {doc['eff_from']}"] if doc.get("eff_from") else [])
        dieu = [f"Điều {n} ({h})" if h else f"Điều {n}" for n, h in list(laws[k]["dieu"].items())[:MAX_DIEU]]
        lines.append(f"{i}. {k} - {laws[k]['title']} - {', '.join(bits)}" + (f"\n   đã lấy: {'; '.join(dieu)}" if dieu else ""))
        if toc is not None and (t := toc(k)):
            lines.append("   mục lục: " + ", ".join(f"Đ{n}" for n, _ in t[:40]))
    fams = [s for s in signals if s["kind"] == "same_family"]
    if fams:
        lines.append("Cùng họ văn bản trong kết quả: " + "; ".join(" ↔ ".join(s["laws"]) for s in fams) + " (không ưu tiên bản nào)")
    for s in signals:
        if s["kind"] == "law_missing":
            lines.append(f"Kiểm tra: văn bản {s['law']} nêu trong câu hỏi chưa có trong kết quả")
        elif s["kind"] == "dieu_missing":
            lines.append(f"Kiểm tra: Điều {s['dieu']} của {s['law']} chưa có trong kết quả")
        elif s["kind"] == "flag_missing":
            lines.append(f"Kiểm tra: câu hỏi nhắc tới {s['flag']} nhưng kết quả không có văn bản loại đó")
    return "\n".join(lines), signals
