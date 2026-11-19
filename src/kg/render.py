"""Text the model reads after a KG tool call: one short line per document, per hop, with the TRUE count when a hop is cut."""


def render_related(key: str, hops: list[dict] | None, title_chars: int = 80) -> str:
    if hops is None:
        return f"Không tìm thấy văn bản {key} trong cơ sở dữ liệu."
    if not hops:
        return f"Không có văn bản sửa đổi / thay thế / bãi bỏ liên quan tới {key}."
    out = []
    for h in hops:
        shown = len(h["docs"])
        out.append(f"hop {h['hop']}: {h['total']} văn bản" + (f", hiện {shown}" if shown < h["total"] else ""))
        out += [f"- {d['key'] or d['id']} ({d['family']}, {d['status'] or 'không rõ'}): {d['title'][:title_chars]}" for d in h["docs"]]
    return "\n".join(out)
