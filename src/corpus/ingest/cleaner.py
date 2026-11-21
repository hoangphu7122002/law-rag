"""LawCleaner: LlamaIndex TransformComponent that cleans law Documents before node parsing.

Course file layout (docs/report/corpus-eda.md): body | "--- THÔNG TIN THÊM ---" (Ten, So hieu, Loai van ban,
Linh vuc nganh, Noi ban hanh, Ngay ban hanh, Ngay hieu luc, Tinh trang) | "--- MỤC LỤC ---" (optional table of
contents) | "--- VĂN BẢN LIÊN QUAN ---" (relation blocks with "(ID: n)" items).
Body noise: Word-export hard wraps (blank line = paragraph); header (issuer, quốc hiệu, motto, Số, date,
separators) before the type + title; "Nơi nhận" + signature after the text (attached rules / appendices that
follow are kept); blank-form placeholders "......".
vbpl / web files (VbplImporter, LuatVietnam): first line = full title, one paragraph per line, no tail blocks.

Out: text = one paragraph per line; metadata["title"] (e.g. "Chỉ thị 06/2008/CT-UBND tăng cường …") goes into
both the embedding and the LLM context through the node templates; number, type, field, issuer, issued,
effective, status, level, year are kept for filters and boosts, hidden from embedding and LLM.

    IngestionPipeline(transformations=[LawCleaner(), DieuNodeParser()]).run(documents=docs)
    LawCleaner.parse(raw_text, source)  -> CleanDoc(text, meta, relations)   # also used by the graph builder
"""
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from llama_index.core.schema import BaseNode, Document, TransformComponent

from corpus.ingest.patterns import (
    BOILER,
    DATE,
    FORM_FILL,
    HEADER_SCAN,
    INFO_KEYS,
    ISSUER_LINE,
    LOCAL,
    NOI_NHAN,
    NUMBER,
    REL_HEADERS,
    REL_ITEM,
    REL_START,
    RESUME,
    SIGN_TITLE,
    SIGNATURE_MAX,
    TAIL,
    TYPE_LINE,
)

SHOWN = ["title"]   # metadata keys visible to embedding and LLM


@dataclass
class CleanDoc:
    text: str
    meta: dict = field(default_factory=dict)
    relations: list[dict] = field(default_factory=list)  # {rel, title, target_id}


class LawCleaner(TransformComponent):
    @classmethod
    def class_name(cls) -> str:
        return "LawCleaner"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> list[BaseNode]:
        return [self.clean_document(n) if isinstance(n, Document) else n for n in nodes]

    def clean_document(self, doc: Document) -> Document:
        c = self.parse(doc.get_content(), doc.metadata.get("source", "corpus"))
        meta = {**doc.metadata, **c.meta}
        hidden = [k for k in meta if k not in SHOWN]
        return Document(text=c.text, id_=doc.id_, metadata=meta, excluded_embed_metadata_keys=hidden,
                        excluded_llm_metadata_keys=hidden, metadata_template="{value}", text_template="{metadata_str}\n{content}")

    # ---- pure parsing -----------------------------------------------------------------------------------------
    @classmethod
    def parse(cls, raw: str, source: str = "corpus") -> CleanDoc:
        text = unicodedata.normalize("NFC", raw).replace("\xa0", " ").replace("\r", "")
        body, info, relations = cls.split_tail(text)
        paras = cls.paragraphs(body)
        first = paras.pop(0) if (source.startswith(("vbpl", "web"))) and paras else None
        head, paras = cls.strip_header(paras, mixed_case_title=first is not None)
        paras = cls.strip_signatures(paras)
        paras = [p for p in paras if not BOILER.match(p) and not cls._is_form(p)]
        meta = {k: v for k, v in head.items() if k in ("number", "year", "level")} | info
        if title := re.sub(r"\s+", " ", first or info.get("title") or cls._header_title(head)).strip():
            meta["title"] = title
        meta.setdefault("level", "local" if LOCAL.search(meta.get("issuer", "")) else "central")
        if "year" not in meta and (y := re.search(r"(?:19|20)\d{2}", meta.get("issued", ""))):
            meta["year"] = int(y.group(0))
        return CleanDoc("\n".join(paras), meta, relations)

    @staticmethod
    def split_tail(text: str) -> tuple[str, dict, list[dict]]:
        m = TAIL.search(text)
        if not m:
            return text, {}, []
        info, rels, block, rel = {}, [], None, None
        for line in TAIL.sub(lambda t: f"\n{t.group(0)}\n", text[m.start():]).splitlines():
            if t := TAIL.search(line):
                block = t.group(1)
            elif block == "THÔNG TIN THÊM" and ":" in line:
                key, val = (s.strip() for s in line.split(":", 1))
                if key in INFO_KEYS and val and not val.startswith("Dữ liệu đang cập nhật"):
                    info[INFO_KEYS[key]] = val
            elif block == "VĂN BẢN LIÊN QUAN":
                if head := REL_START.match(line.strip()):
                    rel = REL_HEADERS[head.group(1)]
                elif (item := REL_ITEM.match(line)) and rel:
                    rels.append({"rel": rel, "title": item.group(1), "target_id": int(item.group(2))})
        return text[: m.start()], info, rels

    @staticmethod
    def paragraphs(text: str) -> list[str]:
        """Blank-line paragraphs (course Word export) are unwrapped; files without blank lines keep one line each."""
        blank = len(re.findall(r"\n\s*\n", text))
        word_export = blank > 0.2 * max(1, text.count("\n") - blank)   # vbpl: one blank line after the title only
        blocks = re.split(r"\n\s*\n", text) if word_export else text.split("\n")
        return [p for p in (re.sub(r"\s+", " ", b).strip() for b in blocks) if p]

    @classmethod
    def strip_header(cls, paras: list[str], mixed_case_title: bool = False) -> tuple[dict, list[str]]:
        """Drop everything before the type line, the title after it, and an issuer line ("CHỦ TỊCH UBND …")."""
        meta: dict = {}
        head = paras[:HEADER_SCAN]
        type_at = next((i for i, p in enumerate(head) if TYPE_LINE.match(p) and cls._mostly_upper(p)), None)
        for p in head[: HEADER_SCAN if type_at is None else type_at]:
            if (n := NUMBER.match(p)) and "number" not in meta:
                meta["number"] = n.group(1).rstrip(".")
            if (d := DATE.search(p)) and "year" not in meta:
                meta["year"] = int(d.group(3))
        if type_at is None:
            return meta, [p for p in paras if not NUMBER.match(p)]
        meta["level"] = "local" if LOCAL.search(" ".join(head[:type_at])) else "central"
        t = TYPE_LINE.match(paras[type_at])
        meta["type_upper"] = t.group(1)
        title = [t.group(2).strip(" :")] if t.group(2).strip(" :") else []
        i = type_at + 1
        while (i < len(paras) and len(title) < 4 and not RESUME.match(paras[i]) and not BOILER.match(paras[i])
               and not ISSUER_LINE.match(paras[i]) and not paras[i].startswith("Căn cứ") and len(paras[i]) < 400
               and (cls._mostly_upper(paras[i]) or mixed_case_title)):
            title.append(paras[i])
            i += 1
        meta["header_title"] = " ".join(title)
        if i < len(paras) and ISSUER_LINE.match(paras[i]) and cls._mostly_upper(paras[i]):
            i += 1
        return meta, paras[i:]

    @classmethod
    def strip_signatures(cls, paras: list[str]) -> list[str]:
        """Drop each "Nơi nhận" block (until attached content resumes) and the signature before it / at the end."""
        out, i = [], 0
        while i < len(paras):
            if NOI_NHAN.match(paras[i]):
                cls._pop_signature(out)
                j = i + 1
                while j < len(paras) and j - i <= SIGNATURE_MAX and not cls._resumes(paras[j]):
                    j += 1
                i = j
                continue
            out.append(paras[i])
            i += 1
        cls._pop_signature(out)
        return out

    @classmethod
    def _resumes(cls, p: str) -> bool:
        """Attached content starts: a structural heading, or an all-caps heading that is not a signer's title."""
        return bool(RESUME.match(p)) or (len(p) >= 8 and cls._mostly_upper(p) and not SIGN_TITLE.match(p)
                                        and not BOILER.match(p))

    @classmethod
    def _pop_signature(cls, out: list[str]) -> None:
        """Signature = up to 6 short trailing paragraphs: titles (TM., CHỦ TỊCH …) and the signer's name."""
        k = 0
        while out and k < 6 and len(out[-1]) < 80 and not out[-1].rstrip().endswith((".", ";", ":")) and (
                SIGN_TITLE.match(out[-1]) or BOILER.match(out[-1]) or cls._is_name(out[-1])):
            out.pop()
            k += 1

    @staticmethod
    def _header_title(head: dict) -> str:
        if not head.get("header_title"):
            return ""
        num = f" {head['number']}" if head.get("number") else ""
        return f"{head.get('type_upper', '').capitalize()}{num}: {head['header_title']}"

    @staticmethod
    def _is_form(p: str) -> bool:
        return sum(len(m) for m in FORM_FILL.findall(p)) > 0.3 * len(p)

    @staticmethod
    def _is_name(p: str) -> bool:
        words = p.split()
        return 2 <= len(words) <= 5 and all(w[:1].isupper() and (w[1:].islower() or len(w) == 1) for w in words)

    @staticmethod
    def _mostly_upper(p: str) -> bool:
        letters = [c for c in p if c.isalpha()]
        return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8
