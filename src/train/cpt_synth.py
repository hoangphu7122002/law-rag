"""CPT augmentation (docs/research: knowledge must be seen in several surface forms to be extractable): for every law article a teacher LLM
(DeepSeek via vibe) writes paraphrases, forward QA (what does Điều N say) and reverse QA (which Điều says this). Pure helpers; the call loop is
scripts/cpt_synth.py. Dev/test gold articles are never sent to the teacher (same rule as the synthetic SFT rows).

    row = {"text", "source": paraphrase | qa_fwd | qa_rev, "key": "<law>#<Điều>"}
Checks per generated text: mentions "Điều N" of its article, introduces no number absent from the article, is not a near copy, paraphrase length
within 0.35-2x, no 13-gram shared with dev/test questions or references.
"""
import json
import re
from dataclasses import dataclass

from train.synth_gen import grams, norm, parse_json_list

NUM = re.compile(r"\d+(?:[.,]\d+)*")
KINDS = ("paraphrase", "qa_fwd", "qa_rev")


@dataclass(frozen=True)
class Article:
    key: str      # "<law key>#<Điều>"
    title: str    # law title line
    text: str     # starts with the "Điều N." heading

    @property
    def law(self) -> str:
        return self.key.rsplit("#", 1)[0]

    @property
    def dieu(self) -> str:
        return self.key.rsplit("#", 1)[1]


@dataclass(frozen=True)
class Batch:
    id: str
    items: tuple


def batches(arts: list[Article], per_call: int = 6, prefix: str = "b") -> list[Batch]:
    return [Batch(f"{prefix}{i // per_call}", tuple(arts[i:i + per_call])) for i in range(0, len(arts), per_call)]


def make_prompt(batch: Batch) -> str:
    head = (
        "Bạn là biên tập viên pháp lý. Với MỖI điều luật dưới đây, hãy viết dữ liệu học cho một mô hình ngôn ngữ nhỏ.\n"
        "Trả về DUY NHẤT một danh sách JSON, mỗi phần tử có dạng:\n"
        '{"id": "<id>", "paraphrases": [3 chuỗi], "qa_fwd": [2 đối tượng {"q","a"}], "qa_rev": [1 đối tượng {"q","a"}]}\n'
        "QUY TẮC (bắt buộc):\n"
        "- paraphrases: 3 bản viết lại KHÁC NHAU về cách diễn đạt và cấu trúc (một bản giải thích dễ hiểu, một bản tóm tắt gạch đầu dòng, một bản văn phong hành chính "
        "đổi thứ tự trình bày). Mỗi bản phải nêu rõ \"Điều N\" và tên luật, giữ NGUYÊN mọi con số, thời hạn, tỷ lệ, tên cơ quan; KHÔNG thêm con số hay quy định không có trong điều luật.\n"
        "- qa_fwd: 2 câu hỏi về nội dung cụ thể của điều; câu trả lời mở đầu bằng \"Căn cứ theo Điều N <tên luật>\" và đúng với điều luật.\n"
        "- qa_rev: 1 câu hỏi mô tả một chi tiết đặc trưng của điều mà KHÔNG nêu số điều, hỏi quy định đó nằm ở điều nào, luật nào; "
        "câu trả lời nêu rõ \"Điều N\" và tên luật (kèm số hiệu).\n"
        "- Không chép nguyên văn cả điều trong paraphrases; không thêm lời giải thích ngoài JSON.\n\nCÁC ĐIỀU LUẬT:\n")
    body = []
    for i, a in enumerate(batch.items):
        body.append(f"### id={i} | {a.key} | {a.title}\n{a.text}\n")
    return head + "\n".join(body)


def numbers(s: str) -> set[str]:
    return {m.replace(",", ".") for m in NUM.findall(s or "")}


def verify_text(text: str, art: Article, kind: str, blocked: frozenset | set = frozenset()) -> str | None:
    t = norm(text)
    if len(t) < 40:
        return "too_short"
    if not re.search(rf"Điều\s+{re.escape(art.dieu)}(?![\w])", t, flags=re.I):
        return "wrong_article" if re.search(r"Điều\s+\d+", t) else "no_article_number"
    if numbers(t) - numbers(art.text) - numbers(art.title) - numbers(art.key):
        return "new_numbers"
    if kind == "paraphrase":
        ratio = len(t) / max(len(norm(art.text)), 1)
        if not 0.35 <= ratio <= 2.0:
            return "length_ratio"
        g, gs = grams(t, 8), grams(art.text, 8)
        if gs and len(g & gs) / max(len(g), 1) > 0.8:
            return "near_copy"
    if kind == "qa_rev" and art.law.lower() not in t.lower().replace(" ", ""):
        return "no_law_number"
    if blocked and grams(t) & blocked:
        return "near_dup_eval"
    return None


def process(raw: str, batch: Batch, blocked=frozenset()):
    """-> (rows, rejects)."""
    objs = parse_json_list(raw)
    if objs is None:
        return [], [{"batch": batch.id, "reason": "bad_json", "raw": (raw or "")[:300]}]
    rows, rej = [], []
    for o in objs:
        try:
            art = batch.items[int(o["id"])]
        except (KeyError, ValueError, IndexError, TypeError):
            rej.append({"batch": batch.id, "reason": "unknown_id"})
            continue
        cands = [("paraphrase", p) for p in o.get("paraphrases") or [] if isinstance(p, str)]
        for kind in ("qa_fwd", "qa_rev"):
            cands += [(kind, f"Hỏi: {norm(x['q'])}\nĐáp: {norm(x['a'])}") for x in o.get(kind) or [] if isinstance(x, dict) and "q" in x and "a" in x]
        for kind, text in cands:
            why = verify_text(text, art, kind, blocked)
            if why:
                rej.append({"batch": batch.id, "key": art.key, "kind": kind, "reason": why})
            else:
                rows.append({"text": text, "source": kind, "key": art.key})
    return rows, rej
