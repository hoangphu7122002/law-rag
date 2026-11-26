"""External CPT text from open-licence Hugging Face datasets (no gated / unlicensed sources), decontaminated against dev/test.

  laws    : undertheseanlp/UTS_VLC (MIT) + tmquan/thuvienphapluat-vn-vbpl (CC-BY-4.0; Luật / Bộ luật / Nghị định / Thông tư / Pháp lệnh / Nghị quyết)
  general : HuggingFaceFW/fineweb-2 vie_Latn (ODC-By), replay against forgetting
Every text is cut into ~1500-char chunks; a chunk that shares a 13-gram with a dev/test question or reference is dropped.
Output rows: {"text", "source"}.   PYTHONPATH=src python3 scripts/data/cpt_external.py --law-chars 60e6 --general-chars 30e6
"""
import argparse
import json
from pathlib import Path

from train.synth import synth_gen as sg

LAW_TYPES = {"Luật", "Bộ luật", "Nghị định", "Thông tư", "Pháp lệnh", "Nghị quyết"}


def chunks(text: str, size: int = 1500):
    buf = ""
    for para in text.split("\n"):
        if buf and len(buf) + len(para) > size:
            yield buf.strip()
            buf = ""
        buf += para + "\n"
    if buf.strip():
        yield buf.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/cpt/ext.jsonl")
    ap.add_argument("--law-chars", type=float, default=60e6)
    ap.add_argument("--general-chars", type=float, default=30e6)
    ap.add_argument("--shards", type=int, default=96)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    _, _, blocked = sg.load_eval_filters(a.processed)
    stats = {"law": [0, 0], "general": [0, 0]}   # kept chars, dropped chunks
    out = open(a.out, "w", encoding="utf-8")

    def emit(text: str, src: str, kind: str, budget: float) -> bool:
        for c in chunks(text):
            if stats[kind][0] >= budget:
                return False
            if len(c) < 200:
                continue
            if sg.grams(c) & blocked:
                stats[kind][1] += 1
                continue
            out.write(json.dumps({"text": c, "source": src}, ensure_ascii=False) + "\n")
            stats[kind][0] += len(c)
        return True

    p = hf_hub_download("undertheseanlp/UTS_VLC", "data/2026-00000-of-00001.parquet", repo_type="dataset")
    for r in pq.read_table(p, columns=["content"]).to_pylist():
        emit(r["content"], "uts_vlc", "law", a.law_chars)
    print("uts_vlc", stats, flush=True)
    for i in range(a.shards):
        if stats["law"][0] >= a.law_chars:
            break
        p = hf_hub_download("tmquan/thuvienphapluat-vn-vbpl", f"documents-{i:05d}-of-00096.parquet", repo_type="dataset")
        for r in pq.read_table(p, columns=["doc_type", "vn_text", "is_english"]).to_pylist():
            if r["doc_type"] in LAW_TYPES and r["vn_text"] and r["is_english"] != "True":
                emit(r["vn_text"], "tvpl_vbpl", "law", a.law_chars)
        Path(p).unlink(missing_ok=True)   # keep the box disk free
        print("shard", i, stats, flush=True)

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-2", "vie_Latn", split="train", streaming=True)
    for r in ds:
        if not emit(r["text"], "fineweb2_vie", "general", a.general_chars):
            break
    print("final", stats, flush=True)


if __name__ == "__main__":
    main()
