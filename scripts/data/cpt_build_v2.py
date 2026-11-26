"""Curated external CPT text (v2): open-licence legal sources + replay, cleaned, deduplicated, PII-masked, decontaminated against dev/test.

Per document: strip HTML -> normalize -> quality filter -> (judgments) PII mask -> Dedup (document, long paragraphs, document number) -> chunks of <=3000 chars
with the title as first line -> drop chunks sharing a 13-gram with a dev/test question or reference. Sources are read file by file into a temp dir that is
removed straight after (disk), in random order, until each token budget (chars / 3.4) is reached. Output rows {"text","source"} + a per-source report.

    PYTHONPATH=src .venv/bin/python scripts/data/cpt_build_v2.py --out data/cpt/ext_v2.jsonl
Licences: UTS_VLC MIT; tmquan/* and th1nhng0 CC-BY-4.0; wikisource/wikipedia CC-BY-SA; FineWeb-2 ODC-By.  Nothing gated, nothing without a stated licence.
"""
import argparse
import collections
import json
import random
import re
import shutil
import time
from pathlib import Path

from train import synth_gen as sg
from train.cpt_clean import Dedup, chunk_doc, mask_pii, normalize, quality, strip_html

CPT = 3.4   # chars per token (measured on our data)
LAW_TYPES = {"Luật", "Bộ luật", "Nghị định", "Thông tư", "Pháp lệnh", "Nghị quyết"}
WS_LAW = re.compile(r"Hiến pháp|Luật|Bộ luật|Nghị định|Thông tư|Pháp lệnh|Nghị quyết|Công ước|Tuyên ngôn|Sắc lệnh", re.I)


class Sink:
    def __init__(self, out, blocked, dedup):
        self.f, self.blocked, self.dedup = out, blocked, dedup
        self.stats = collections.defaultdict(lambda: collections.Counter())

    def doc(self, source, text, title="", key=None, pii=False, html_in=False, min_chars=300) -> int:
        st = self.stats[source]
        st["docs_seen"] += 1
        t = normalize(strip_html(text) if html_in else text)
        why = quality(t, min_chars)
        if why:
            st["drop_" + why] += 1
            return 0
        if pii:
            t = mask_pii(t)
        t = self.dedup.accept(f"{source}", t, key)
        if t is None:
            st["drop_duplicate"] += 1
            return 0
        n = 0
        for c in chunk_doc(t, title=title.strip()):
            if sg.grams(c) & self.blocked:
                st["drop_chunk_eval_overlap"] += 1
                continue
            self.f.write(json.dumps({"text": c, "source": source}, ensure_ascii=False) + "\n")
            n += len(c)
        st["docs_kept"] += 1 if n else 0
        st["chars"] += n
        return n


def download(repo, path, tmp):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", local_dir=str(tmp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cpt/ext_v2.jsonl")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--tmp", default="data/cpt/_dl")
    ap.add_argument("--budget", default="uts=6e6,phapdien=41e6,anle=15e6,th=40e6,tvpl=25e6,cbba=30e6,wikisource=3e6,wikipedia=4e6,fineweb=8e6", help="tokens per source")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-free-gb", type=int, default=40)
    ap.add_argument("--only", default="", help="comma list of sources (smoke tests)")
    a = ap.parse_args()
    import pyarrow.parquet as pq

    budget = {k: float(v) * CPT for k, v in (x.split("=") for x in a.budget.split(","))}
    only = set(a.only.split(",")) if a.only else set(budget)
    rng = random.Random(a.seed)
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    _, _, blocked = sg.load_eval_filters(a.processed)
    sink = Sink(open(a.out, "w", encoding="utf-8"), blocked, Dedup())

    def free_gb():
        return shutil.disk_usage(tmp).free / 1e9

    def run(name, repo, files, rows_of, cols):
        """files: list of repo paths (random order); rows_of(table_batch) -> iterable of kwargs for sink.doc."""
        done = 0
        for path in files:
            if done >= budget[name]:
                break
            if free_gb() < a.min_free_gb:
                print("disk guard: stop", name, flush=True)
                break
            p = download(repo, path, tmp)
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=1000, columns=cols):
                for r in batch.to_pylist():
                    for kw in rows_of(r):
                        done += sink.doc(name, **kw)
                    if done >= budget[name]:
                        break
                if done >= budget[name]:
                    break
            Path(p).unlink(missing_ok=True)
            print(f"{name}: {path} -> {done / CPT / 1e6:.1f}M tokens so far {dict(sink.stats[name])}", flush=True)

    def sel(names):
        return [n for n in names if n in only]

    if "uts" in only:
        run("uts", "undertheseanlp/UTS_VLC", [f"data/{s}-00000-of-00001.parquet" for s in ("2026", "2026_01", "2023", "2021")],
            lambda r: [dict(text=r["content"], title=r.get("title") or "", key=r.get("id"))], ["id", "title", "content"])
    if "phapdien" in only:
        run("phapdien", "tmquan/phapdien-moj-gov-vn", [f"documents-{i:05d}-of-00007.parquet" for i in range(7)],
            lambda r: [dict(text=r["content_text"] or "", title=f'Bộ Pháp điển: {r["subject_title_vi"] or ""} - {r["article_title"] or ""}', min_chars=150)],
            ["subject_title_vi", "article_title", "content_text"])
    if "anle" in only:
        run("anle", "tmquan/anle-toaan-gov-vn", ["documents-00000-of-00001.parquet"],
            lambda r: [dict(text=r["markdown"] or "", title="" if (r["doc_name"] or "").isdigit() else (r["doc_name"] or ""), pii=True, min_chars=1500)], ["doc_name", "markdown"])
    if "th" in only:   # vbpl.vn laws (HTML) joined with the metadata for the title and the document number
        meta = {r["id"]: r for r in pq.read_table(download("th1nhng0/vietnamese-legal-documents", "data/metadata.parquet", tmp),
                                                  columns=["id", "so_ky_hieu", "title", "loai_van_ban"]).to_pylist()}
        def th_rows(r):
            m = meta.get(r["id"], {})
            if m.get("loai_van_ban") and m["loai_van_ban"] not in LAW_TYPES:
                return []
            return [dict(text=r["content_html"] or "", title=m.get("title") or "", key=m.get("so_ky_hieu"), html_in=True)]
        pth = download("th1nhng0/vietnamese-legal-documents", "data/content.parquet", tmp)
        pf = pq.ParquetFile(pth)
        order = list(range(pf.metadata.num_row_groups))
        rng.shuffle(order)
        done = 0
        for g in order:
            if done >= budget["th"] or free_gb() < a.min_free_gb:
                break
            for r in pf.read_row_group(g, columns=["id", "content_html"]).to_pylist():
                for kw in th_rows(r):
                    done += sink.doc("th", **kw)
                if done >= budget["th"]:
                    break
            print(f"th: row group {g} -> {done / CPT / 1e6:.1f}M tokens {dict(sink.stats['th'])}", flush=True)
        Path(pth).unlink(missing_ok=True)
    if "tvpl" in only:
        shards = [f"documents-{i:05d}-of-00096.parquet" for i in range(95)]
        rng.shuffle(shards)
        run("tvpl", "tmquan/thuvienphapluat-vn-vbpl", shards,
            lambda r: [dict(text=r["vn_text"] or "", key=(f'{r["doc_type"]} {r["doc_number"]} {r["year"]}' if r["doc_number"] not in (None, "None") else None))]
            if r["doc_type"] in LAW_TYPES and r["is_english"] != "True" else [], ["doc_type", "doc_number", "year", "is_english", "vn_text"])
    if "cbba" in only:   # sampled judgments: 5% of the rows of 2 random shards (100k rows each), PII masked
        shards = [f"documents-{i:05d}-of-00016.parquet" for i in range(16)]
        rng.shuffle(shards)
        run("cbba", "tmquan/cbba-toaan-gov-vn", shards[:3],
            lambda r: [dict(text=r["markdown"] or "", title="" if (r["doc_name"] or "").isdigit() else (r["doc_name"] or ""), pii=True, min_chars=2000)] if rng.random() < 0.05 else [], ["doc_name", "markdown"])
    if "wikisource" in only:
        run("wikisource", "wikimedia/wikisource", ["20231201.vi/train-00000-of-00001.parquet"],
            lambda r: [dict(text=r["text"], title=r["title"])] if WS_LAW.search(r["title"] or "") else [], ["title", "text"])
    if "wikipedia" in only:
        from huggingface_hub import list_repo_files
        files = [f for f in list_repo_files("wikimedia/wikipedia", repo_type="dataset") if f.startswith("20231101.vi/")]
        rng.shuffle(files)
        run("wikipedia", "wikimedia/wikipedia", files, lambda r: [dict(text=r["text"], title=r["title"], min_chars=800)] if rng.random() < 0.15 else [], ["title", "text"])
    if "fineweb" in only:
        from datasets import load_dataset
        done = 0
        for r in load_dataset("HuggingFaceFW/fineweb-2", "vie_Latn", split="train", streaming=True).shuffle(seed=a.seed, buffer_size=20000):
            done += sink.doc("fineweb", r["text"], min_chars=600)
            if done >= budget["fineweb"]:
                break
        print("fineweb:", dict(sink.stats["fineweb"]), flush=True)
    sink.f.close()
    rep = {k: {**dict(v), "tokens_est_M": round(v["chars"] / CPT / 1e6, 2)} for k, v in sink.stats.items()}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    import os
    os._exit(0)   # a streaming dataset thread aborts the interpreter at shutdown


if __name__ == "__main__":
    main()
