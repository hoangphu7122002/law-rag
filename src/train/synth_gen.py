"""Synthetic Q/A generator core (teacher-data-plan Stage 3b). Pure: pool, plan, prompt, verify, budget helpers.

A teacher LLM writes NEW question + template answer pairs from corpus articles. Articles that are gold of any dev / test
question never enter the pool; dev / test questions and references are used only as filters (never sent anywhere).
CLI: scripts/synth_gen.py.
"""
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from corpus.decontam import grams
from corpus.keys import stem_key

HEADING = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zđ]?)\s*[.:]")
MIN_CHARS, MAX_CHARS = 200, 6000
DEFAULT_MIX = dict(single=40, multi_same_law=20, multi_law=10, version_family=10, vague=10, no_answer=10)
PATTERNS = tuple(DEFAULT_MIX)
Q_LEN, A_LEN, REFUSAL_MIN = (40, 400), (100, 4000), 40
QUOTE_MIN = 0.6
REFUSAL = "Dựa trên các ngữ cảnh được cung cấp, chưa tìm thấy quy định liên quan để trả lời câu hỏi này."
REFUSAL_MARKS = ("chưa tìm thấy", "không tìm thấy", "không đề cập", "không có quy định", "không quy định")


@dataclass(frozen=True)
class Article:
    law: str      # law key (corpus.keys)
    dieu: str
    text: str     # starts with the "Điều N." heading
    stem: str = ""

    @property
    def key(self) -> str:
        return f"{self.law}#{self.dieu}"

    @property
    def heading(self) -> str:
        return re.sub(r"^\s*Điều\s+\S+?\s*[.:]\s*", "", self.text.split("\n", 1)[0]).strip()


@dataclass(frozen=True)
class Item:
    iid: str
    articles: tuple


@dataclass(frozen=True)
class Batch:
    id: str
    pattern: str
    items: tuple


def split_articles(text: str) -> list[tuple[str, str]]:
    """[(Điều number, body incl. heading line)] from one document text."""
    ms = list(HEADING.finditer(text))
    out = []
    for m, nxt in zip(ms, [*ms[1:], None]):
        out.append((m.group(1).lower(), text[m.start():nxt.start() if nxt else len(text)].strip()))
    return out


def build_pool(dirs, excluded: set, per_law: int = 5, seed: int = 0) -> list[Article]:
    by_law: dict[str, dict[str, Article]] = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.txt")):
            law = stem_key(f.stem)
            for dieu, body in split_articles(f.read_text(errors="ignore")):
                a = Article(law, dieu, body, f.stem)
                if a.key in excluded or not MIN_CHARS <= len(body) <= MAX_CHARS:
                    continue
                by_law.setdefault(law, {}).setdefault(dieu, a)
    rng = random.Random(f"{seed}-pool")
    pool = []
    for law in sorted(by_law):
        arts = sorted(by_law[law].values(), key=lambda a: a.dieu)
        rng.shuffle(arts)
        pool += arts[:per_law]
    return pool


def mix_counts(n: int, mix: dict) -> dict:
    tot = sum(mix.values())
    raw = {k: n * v / tot for k, v in mix.items()}
    out = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def _topic_sim(a: str, b: str) -> float:
    x, y = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(x & y) / len(x | y) if x | y else 0.0


def _items(pattern, n, pool, rng, family_of):
    by_law: dict[str, list] = {}
    for a in pool:
        by_law.setdefault(a.law, []).append(a)
    combos, seen = [], set()

    def add(arts):
        k = frozenset(a.key for a in arts)
        if k not in seen:
            seen.add(k)
            combos.append(tuple(arts))

    if pattern in ("single", "vague", "no_answer"):
        for a in rng.sample(pool, min(n, len(pool))):
            add([a])
    elif pattern == "multi_same_law":
        laws = sorted(l for l, v in by_law.items() if len(v) >= 2)
        for _ in range(n * 5 if laws else 0):
            if len(combos) >= n:
                break
            v = by_law[rng.choice(laws)]
            add(rng.sample(v, rng.randint(2, min(3, len(v)))))
    elif pattern == "multi_law":
        laws = sorted(by_law)
        for _ in range(n * 5 if len(laws) >= 2 else 0):
            if len(combos) >= n:
                break
            l1, l2 = rng.sample(laws, 2)
            add([rng.choice(by_law[l1]), rng.choice(by_law[l2])])
    elif pattern == "version_family" and family_of:
        cands = []
        for l1 in sorted(by_law):
            for l2 in sorted(family_of(l1) & by_law.keys()):
                if l2 > l1:
                    cands += [(a, b) for a in by_law[l1] for b in by_law[l2]
                              if a.dieu == b.dieu and _topic_sim(a.heading, b.heading) >= 0.5]
        rng.shuffle(cands)
        for c in cands[:n]:
            add(list(c))
    return combos[:n]


def plan(pool, n_items: int, mix: dict, seed: int, per_call: int = 10, family_of=None) -> list[Batch]:
    batches = []
    for pat, n in mix_counts(n_items, mix).items():
        rng = random.Random(f"{seed}-{pat}")
        combos = _items(pat, n, pool, rng, family_of)
        for i in range(0, len(combos), per_call):
            items = tuple(Item(f"q{k + 1}", c) for k, c in enumerate(combos[i:i + per_call]))
            batches.append(Batch(f"{pat}-{seed}-{i // per_call}", pat, items))
    return batches


def law_kind(key: str) -> str:
    toks = set(re.split(r"[-/]", key))
    for t, name in (("TTLT", "Thông tư liên tịch"), ("NĐ", "Nghị định"), ("QĐ", "Quyết định"), ("NQ", "Nghị quyết"),
                    ("TT", "Thông tư"), ("L", "Luật")):
        if t in toks or (t == "L" and any(x.startswith("QH") for x in toks)):
            return name
    return "Văn bản"


RULES = {
    "single": "Mỗi nguồn là MỘT điều luật. Viết 1 câu hỏi pháp lý cụ thể mà điều đó trả lời được.",
    "multi_same_law": "Mỗi nguồn gồm 2-3 điều của CÙNG một văn bản. Viết 1 câu hỏi cần kết hợp các điều đó để trả lời.",
    "multi_law": "Mỗi nguồn gồm 2 điều của 2 văn bản KHÁC nhau. Viết 1 câu hỏi cần cả hai văn bản để trả lời.",
    "version_family": "Mỗi nguồn gồm 2 điều cùng số, cùng chủ đề của 2 văn bản cùng họ (bản gốc / bản sửa đổi). "
                      "Viết 1 câu hỏi mà đáp án phụ thuộc vào việc áp dụng văn bản nào; trích cả hai và nêu rõ khác biệt.",
    "vague": "Viết câu hỏi bằng lời thường ngày, mơ hồ hoặc suồng sã (không nêu số điều, không nêu tên văn bản), "
             "nhưng vẫn trả lời được bằng điều luật nguồn.",
    "no_answer": "Viết 1 câu hỏi cùng lĩnh vực nhưng nội dung KHÔNG được điều luật nguồn quy định. "
                 f"Câu trả lời là câu từ chối, ví dụ: \"{REFUSAL}\" (có thể thêm 1 câu giải thích điều luật nguồn không đề cập gì).",
}


def make_prompt(batch: Batch) -> str:
    quote = ("" if batch.pattern == "no_answer" else
             "Câu trả lời PHẢI theo mẫu: \"Căn cứ theo Điều X <Thông tư/Nghị định/Luật + số hiệu> quy định về <chủ đề> như sau:\" "
             "rồi TRÍCH NGUYÊN VĂN (từng chữ, không diễn đạt lại, không bịa) các khoản liên quan trong nguồn. "
             "Với nhiều điều: mỗi điều một đoạn mở đầu như vậy.\n")
    head = (f"Bạn tạo dữ liệu huấn luyện hỏi đáp pháp luật Việt Nam từ các điều luật nguồn bên dưới.\n{RULES[batch.pattern]}\n"
            "Câu hỏi trang trọng, dạng \"Theo quy định ..., ...?\" (40-400 ký tự), tự đứng được, không nhắc \"nguồn\" hay \"đoạn trên\".\n"
            f"{quote}"
            f"Mỗi nguồn cho đúng 1 phần tử. Chỉ in MỘT danh sách JSON hợp lệ, không văn bản khác: "
            "[{\"id\": \"<id nguồn>\", \"question\": \"...\", \"answer\": \"...\"}, ...]. Dùng \\n cho xuống dòng trong chuỗi.\n\n")
    body = []
    for it in batch.items:
        body.append(f"### id: {it.iid}")
        for a in it.articles:
            body.append(f"[Điều {a.dieu} - {law_kind(a.law)} {a.law}]\n{a.text}")
        body.append("")
    return head + "\n".join(body)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _body(answer: str) -> str:
    """Answer without the template opener(s) "Căn cứ ... như sau:"."""
    return re.sub(r"(?m)^\s*Căn cứ[^\n]*?(?:như sau\s*:|:)\s*", "", answer)


def quote_fraction(answer: str, sources: str) -> float:
    src = norm(sources)
    total = found = 0
    for line in _body(answer).split("\n"):
        line = re.sub(r"^\s*[-+•*]\s*", "", norm(line))
        if not line:
            continue
        total += len(line)
        if line in src:
            found += len(line)
            continue
        found += sum(len(p) for p in (norm(x) for x in re.split(r"(?<=[;.:])\s+", line)) if p and p in src)
    return found / total if total else 0.0


def verify(obj: dict, item: Item, pattern: str, eval_questions: set, blocked_grams: set) -> str | None:
    q, a = str(obj.get("question", "")).strip(), str(obj.get("answer", "")).strip()
    if not Q_LEN[0] <= len(q) <= Q_LEN[1]:
        return "question_len"
    lo = REFUSAL_MIN if pattern == "no_answer" else A_LEN[0]
    if not lo <= len(a) <= A_LEN[1]:
        return "answer_len"
    if norm(q).lower() in eval_questions or grams(q) & blocked_grams:
        return "near_dup_eval"
    if pattern == "no_answer":
        return None if any(m in a.lower() for m in REFUSAL_MARKS) else "not_refusal"
    if quote_fraction(a, "\n".join(x.text for x in item.articles)) < QUOTE_MIN:
        return "quote_not_verbatim"
    return None


def parse_json_list(raw: str):
    s = re.sub(r"```(?:json)?", "", raw or "")
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        out = json.loads(s[i:j + 1])
    except ValueError:
        return None
    return out if isinstance(out, list) else None


def process(raw: str, batch: Batch, eval_questions: set, blocked_grams: set, seen: set):
    """-> (ok rows, rejects). `seen` (normalised questions) is updated in place for duplicate removal."""
    objs = parse_json_list(raw)
    if objs is None:
        return [], [{"batch": batch.id, "reason": "bad_json", "raw": (raw or "")[:300]}]
    items = {it.iid: it for it in batch.items}
    rows, rej = [], []
    for o in objs:
        if not isinstance(o, dict):
            rej.append({"batch": batch.id, "reason": "bad_item"})
            continue
        it = items.get(str(o.get("id")))
        why = "unknown_id" if it is None else verify(o, it, batch.pattern, eval_questions, blocked_grams)
        if why is None and norm(o["question"]).lower() in seen:
            why = "duplicate"
        if why:
            rej.append({"batch": batch.id, "reason": why, "id": o.get("id"), "question": o.get("question")})
            continue
        seen.add(norm(o["question"]).lower())
        rows.append({"pattern": batch.pattern, "question": o["question"].strip(), "answer": o["answer"].strip(),
                     "sources": [a.key for a in it.articles], "ok": True, "batch": batch.id, "id": it.iid})
    return rows, rej


def estimate_tokens(prompt: str, n_items: int) -> int:
    return int((len(prompt) + 1200 * n_items) / 3.5)


def within_budget(used: int, add: int, max_tokens: int) -> bool:
    return used + add <= max_tokens


def min_interval(calls_per_minute: float) -> float:
    return 60.0 / calls_per_minute


def read_done(path) -> set:
    p = Path(path)
    return set(p.read_text().split()) if p.exists() else set()


def mark_done(path, batch_id: str) -> None:
    with open(path, "a") as f:
        f.write(batch_id + "\n")


def load_eval_filters(processed=Path("data/processed")):
    """(excluded "law#Dieu" of dev/test gold, normalised dev/test questions, 13-grams of dev/test questions + references).
    Filters only: no test text leaves this process."""
    import pandas as pd

    processed = Path(processed)
    split = json.loads((processed / "split.json").read_text())
    ev = set(split["dev"]) | set(split["test"])
    gold = pd.read_parquet(processed / "gold.parquet")
    excluded = {str(p).lower() for _, r in gold[gold.qid.isin(ev)].iterrows() for p in r["pairs"]}
    excluded = {f"{p.split('#')[0].upper()}#{p.split('#')[1]}" if "#" in p else p for p in excluded}
    qa = pd.read_parquet(processed / "qa.parquet")
    qa = qa[qa.qid.isin(ev)]
    questions = {norm(q).lower() for q in qa.question}
    blocked = set()
    for t in [*qa.question, *qa.answer, *(x for x in qa.urax_answer if isinstance(x, str))]:
        blocked |= grams(t)
    return excluded, questions, blocked
