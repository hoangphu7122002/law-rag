"""Continued-pretraining data: the text of the laws our questions are about, packed into fixed-length token sequences.

    PYTHONPATH=src python -m train.cpt_data --out data/cpt/laws.jsonl --base Qwen/Qwen3.5-2B [--seq-len 2048] [--limit N]

Wanted laws = every document cited by a train / dev / test question (gold.parquet docs); their text comes from the index nodes (the same corpus the
retriever reads, so nothing outside the knowledge base is used). Pure helpers: docs_text, pack.
"""
import argparse
import json
from pathlib import Path


def docs_text(nodes, wanted: set[str]) -> dict[str, str]:
    """nodes: iterable of (doc_key, order, text) -> {doc_key: chunks of that document in order, joined by newlines}."""
    by: dict[str, list] = {}
    for doc, order, text in nodes:
        if doc in wanted:
            by.setdefault(doc, []).append((order, text))
    return {d: "\n".join(t for _, t in sorted(v)) for d, v in by.items()}


def articles(nodes, wanted: set[str]) -> list[dict]:
    """nodes: iterable of (doc_key, article number or None, position, text of one chunk, title line first).
    -> [{key "doc#Điều", title, text}] one entry per article: its chunks in position order, the repeated title line dropped, nodes without an article heading skipped."""
    by: dict = {}
    for doc, art, pos, text in nodes:
        if doc in wanted and art:
            by.setdefault((doc, art), []).append((pos, text))
    out = []
    for (doc, art), chunks in sorted(by.items(), key=lambda kv: (kv[0][0], min(p for p, _ in kv[1]))):
        chunks.sort()
        title = chunks[0][1].split("\n", 1)[0]
        parts = [chunks[0][1].split("\n", 1)[1] if "\n" in chunks[0][1] else chunks[0][1]]
        for _, t in chunks[1:]:
            first, _, rest = t.partition("\n")
            parts.append(rest if first == title and rest else t)
        out.append({"key": f"{doc}#{art}", "title": title, "text": "\n".join(parts)})
    return out


def pack(docs: list[list[int]], seq_len: int, eos: int) -> list[list[int]]:
    """Concatenate token lists with an EOS after each document and cut sequences of exactly seq_len; the incomplete tail is dropped."""
    flat: list[int] = []
    for d in docs:
        flat.extend(d)
        flat.append(eos)
    return [flat[i:i + seq_len] for i in range(0, len(flat) - seq_len + 1, seq_len)]


def main() -> None:
    import polars as pl
    from transformers import AutoTokenizer

    import re

    from llama_index.core.schema import MetadataMode

    from config import load
    from corpus.keys import stem_key
    from evaluate.retrieval_eval import RetrievalEval
    from index import IndexStore

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N documents")
    ap.add_argument("--articles-out", default="", help="also write one jsonl line per article ({key, title, text}) for the augmentation step (train.cpt_synth)")
    a = ap.parse_args()
    wanted = {d for r in pl.read_parquet("data/processed/gold.parquet").iter_rows(named=True) for d in (r["docs"] or ())}
    store = IndexStore(load("configs/exp/A5_v3b.yaml"))
    def order(i, n):   # chunks of one law in article order (the docstore order is not the document order); the whole key sorts (Điều number, position)
        art = RetrievalEval.passage_keys({"doc": n.metadata.get("doc", ""), "text": n.get_content()[:400]})[1]
        m = re.match(r"\d+", art or "")
        return (int(m.group()) if m else 0, i)

    # LLM-mode content keeps the law title line before each chunk, so every article is tied to its law (as in the retrieval prompts)
    nodes = ((stem_key(n.metadata.get("doc", "")), order(i, n), n.get_content(metadata_mode=MetadataMode.LLM)) for i, n in enumerate(store.nodes()))
    if a.articles_out:
        def art_nodes():
            for i, n in enumerate(store.nodes()):
                art = RetrievalEval.passage_keys({"doc": n.metadata.get("doc", ""), "text": n.get_content()[:400]})[1]
                yield stem_key(n.metadata.get("doc", "")), art, i, n.get_content(metadata_mode=MetadataMode.LLM)
        arts = articles(art_nodes(), wanted)
        Path(a.articles_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.articles_out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in arts), encoding="utf-8")
        print(f"{len(arts)} articles -> {a.articles_out}")
    texts = docs_text(nodes, wanted)
    keys = sorted(texts)[: a.limit or None]
    tok = AutoTokenizer.from_pretrained(a.base)
    ids = [tok(texts[k], add_special_tokens=False)["input_ids"] for k in keys]
    seqs = pack(ids, a.seq_len, tok.eos_token_id)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps({"input_ids": s}) + "\n" for s in seqs), encoding="utf-8")
    print(f"{len(keys)}/{len(wanted)} wanted documents found, {sum(len(i) for i in ids) / 1e6:.2f}M tokens, {len(seqs)} sequences of {a.seq_len} -> {a.out}")


if __name__ == "__main__":
    main()
