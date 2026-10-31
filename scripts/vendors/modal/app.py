"""Modal runner for law-rag GPU jobs (1x L4 each).

Data: Volume "law-rag-data" at /data holds Dataset.zip, qa.parquet, split.json (upload with `make modal-data`).
Results: Volume "law-rag-results" at /results/<job>/... ; pull with `modal volume get law-rag-results <job> results/runs/`.

Jobs (each one container, detached, timeout = hard cost cap):
    baseline   B0/B1: the course notebook pipeline (LangChain + vietnamese-sbert + FAISS + Qwen1.5-1.8B) on test + dev
    embed      cache corpus embeddings: 2 chunkings x {bge-m3, Vietnamese_Embedding, multilingual-e5-large}
    models     zero-shot model sweep on dev with vLLM (baseline retrieval), baseline prompt vs "Căn cứ" template
    trainprobe LoRA continued-pretraining throughput (tokens/s) on law text for 1.5B and 3B models

    env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=<p> modal run --detach scripts/vendors/modal/app.py::<job>
"""
import json
import os
import time
from pathlib import Path

import modal

app = modal.App("law-rag")
data_vol = modal.Volume.from_name("law-rag-data", create_if_missing=True)
res_vol = modal.Volume.from_name("law-rag-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("law-rag-hf-cache", create_if_missing=True)
VOLS = {"/data": data_vol, "/results": res_vol, "/root/.cache/huggingface": hf_cache}
_here = Path(__file__).resolve()
# local: <repo>/scripts/vendors/modal/app.py ; in the container the file sits at /root/app.py
SRC = _here.parents[3] / "src" if len(_here.parents) > 3 else Path("/src")

base = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1", "sentence-transformers==3.3.1",
    "polars==1.14.0", "pyarrow", "numpy<2", "tqdm", "scikit-learn",
)
lc_image = base.pip_install(
    "langchain==0.3.9", "langchain-community==0.3.8", "langchain-huggingface==0.1.2",
    "langchain-text-splitters==0.3.2", "faiss-cpu==1.9.0",
).add_local_dir(SRC, "/src")
emb_image = base.pip_install("langchain-text-splitters==0.3.2").add_local_dir(SRC, "/src")
train_image = base.pip_install("peft==0.13.2", "datasets==3.1.0").add_local_dir(SRC, "/src")
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.5", "sentence-transformers==3.3.1", "faiss-cpu==1.9.0", "polars==1.14.0",
                 "pyarrow", "langchain-text-splitters==0.3.2", "scikit-learn")
    .add_local_dir(SRC, "/src")
)


# ---------- shared helpers (run inside containers) ----------

def _corpus(extract_to="/tmp/corpus"):
    import zipfile
    out = Path(extract_to)
    if not out.exists():
        with zipfile.ZipFile("/data/Dataset.zip") as z:
            z.extractall("/tmp/zip")
        out.mkdir(parents=True)
        for p in Path("/tmp/zip").rglob("*.txt"):
            # keep every file even when names differ only by case
            target = out / p.name
            if target.exists():
                target = out / f"{p.stem}__dup{len(list(out.glob(p.stem + '*')))}.txt"
            p.rename(target)
    return sorted(out.glob("*.txt"))


def _qa(split_name):
    import polars as pl
    qa = pl.read_parquet("/data/qa.parquet")
    ids = json.loads(Path("/data/split.json").read_text())[split_name]
    return qa.filter(pl.col("qid").is_in(ids)).sort("qid")


def _save(job, name, obj):
    p = Path("/results") / job / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    else:
        obj.write_parquet(p)
    res_vol.commit()


def _score(rows, embed):
    """rows: list of dicts with pred, answer. Returns per-row metrics + means."""
    import sys
    sys.path.insert(0, "/src")
    from evaluate.metrics import all_lexical, cosine_similarity_score
    out = []
    for r in rows:
        m = all_lexical(r["pred"], r["answer"])
        m["cosine"] = cosine_similarity_score(r["pred"], r["answer"], embed)
        out.append({**r, **m})
    keys = ["cosine", "jaccard", "token_overlap", "bleu1", "rouge_l"]
    means = {k: sum(o[k] for o in out) / max(len(out), 1) for k in keys}
    return out, means


# ---------- J1 baseline (faithful to Baseline.ipynb) ----------

@app.function(image=lc_image, gpu="L4", volumes=VOLS, timeout=4 * 3600)
def baseline():
    import polars as pl
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFacePipeline
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tqdm import tqdm

    job, t0 = "baseline", time.time()
    files = _corpus()
    docs = []
    for f in files:
        for d in TextLoader(str(f), encoding="utf-8").load():
            d.page_content = d.page_content.replace("\n", " ").lower().strip()
            docs.append(d)
    texts = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, length_function=len).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"device": "cuda"})
    vs = None
    for i in tqdm(range(0, len(texts), 1024), desc="embed"):
        part = FAISS.from_documents(texts[i:i + 1024], embeddings)
        vs = part if vs is None else (vs.merge_from(part) or vs)
    vs.save_local("/results/baseline/faiss_db")
    _save(job, "index_info.json", {"files": len(files), "chunks": len(texts), "seconds": time.time() - t0})

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen1.5-1.8B", task="text-generation", device=0,
        model_kwargs={"temperature": 0, "max_length": 1024},
    )

    def rag_query(question, k=5):
        hits = vs.similarity_search(question, k=k)
        context = "\n".join(h.page_content for h in hits)
        prompt = f"""
Bạn là một trợ lý thông minh. Trả lời câu hỏi dựa trên ngữ cảnh sau.
Nếu không đủ thông tin, hãy nói rõ điều đó, không được tự bịa đặt.

Ngữ cảnh:
{context}

Câu hỏi: {question}
Trả lời ngắn gọn và chính xác nhất có thể:
"""
        out = llm.generate([prompt], max_new_tokens=256).generations[0][0].text
        return prompt, out, [h.metadata.get("source", "") for h in hits]

    for split_name in ("test", "dev"):
        rows = []
        for r in tqdm(_qa(split_name).iter_rows(named=True), desc=split_name):
            t = time.time()
            try:
                prompt, raw, sources = rag_query(r["question"])
            except Exception as e:  # keep going, like the notebook
                prompt, raw, sources = "", f"ERROR {e}", []
            answer_only = raw[len(prompt):] if raw.startswith(prompt) else raw
            rows.append({"qid": r["qid"], "question": r["question"], "answer": r["answer"], "law_id": r["law_id"],
                         "raw": raw, "answer_only": answer_only, "prompt_echoed": raw.startswith(prompt),
                         "sources": sources, "seconds": time.time() - t})
            if len(rows) % 25 == 0:
                _save(job, f"{split_name}_partial.parquet", pl.DataFrame(rows))
        emb = embeddings.embed_query
        b0, m0 = _score([{**x, "pred": x["raw"]} for x in rows], emb)
        b1, m1 = _score([{**x, "pred": x["answer_only"]} for x in rows], emb)
        _save(job, f"{split_name}_B0.parquet", pl.DataFrame(b0))
        _save(job, f"{split_name}_B1.parquet", pl.DataFrame(b1))
        _save(job, f"{split_name}_metrics.json", {"n": len(rows), "B0_as_is": m0, "B1_answer_only": m1,
                                                  "prompt_echoed_share": sum(x["prompt_echoed"] for x in rows) / len(rows)})
    _save(job, "DONE.json", {"seconds": time.time() - t0})


# ---------- J2 embedding cache ----------

EMB_MODELS = {
    "bge-m3": ("BAAI/bge-m3", "", ""),
    "vn-emb": ("AITeamVN/Vietnamese_Embedding", "", ""),
    "e5-large": ("intfloat/multilingual-e5-large", "passage: ", "query: "),
}


def _chunks(chunk_size, overlap):
    import polars as pl
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sp = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    rows = []
    for f in _corpus():
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i, c in enumerate(sp.split_text(text)):
            rows.append({"doc": f.stem, "i": i, "text": c})
    return pl.DataFrame(rows)


@app.function(image=emb_image, gpu="L4", volumes=VOLS, timeout=4 * 3600)
def embed(chunkings: str = "c2000:2000:50,c1000:1000:100", models: str = "bge-m3,vn-emb,e5-large"):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    job = "embed"
    for spec in chunkings.split(","):
        name, size, ov = spec.split(":")
        df = _chunks(int(size), int(ov))
        _save(job, f"{name}/chunks.parquet", df)
        for key in models.split(","):
            out = Path("/results") / job / name / f"{key}.npy"
            if out.exists():
                continue
            model_id, doc_prefix, _ = EMB_MODELS[key]
            m = SentenceTransformer(model_id, device="cuda")
            m.half()
            m.max_seq_length = 512
            t = time.time()
            vecs = m.encode([doc_prefix + x for x in df["text"].to_list()], batch_size=128,
                            normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
            np.save(out, vecs.astype(np.float16))
            _save(job, f"{name}/{key}.json", {"model": model_id, "n": len(vecs), "dim": int(vecs.shape[1]),
                                               "seconds": time.time() - t})
            del m
    _save(job, "DONE.json", {"ok": True})


# ---------- J3 model sweep (dev only) ----------

SWEEP = {
    "qwen1.5-1.8b": ("Qwen/Qwen1.5-1.8B", False),
    "qwen2.5-1.5b-it": ("Qwen/Qwen2.5-1.5B-Instruct", True),
    "qwen2.5-3b-it": ("Qwen/Qwen2.5-3B-Instruct", True),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", True),
    "vi-qwen2-3b-rag": ("AITeamVN/Vi-Qwen2-3B-RAG", True),
}
TEMPLATE_SYS = ("Bạn là trợ lý pháp luật Việt Nam. Chỉ dựa vào ngữ cảnh. Trả lời theo mẫu: "
                "\"Căn cứ tại khoản ... Điều ... <số hiệu văn bản> quy định ...\", sau đó trích nguyên văn nội dung "
                "liên quan và giải thích ngắn. Nếu ngữ cảnh không có căn cứ, nói rõ.")


@app.function(image=vllm_image, gpu="L4", volumes=VOLS, timeout=3 * 3600)
def models(names: str = ",".join(SWEEP)):
    import faiss
    import polars as pl
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    from vllm import LLM, SamplingParams

    job = "models"
    sp = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    texts = []
    for f in _corpus():
        t = f.read_text(encoding="utf-8", errors="ignore").replace("\n", " ").lower().strip()
        texts.extend(sp.split_text(t))
    enc = SentenceTransformer("keepitreal/vietnamese-sbert", device="cuda")
    vecs = enc.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    dev = _qa("dev")
    qv = enc.encode(dev["question"].to_list(), normalize_embeddings=True, convert_to_numpy=True)
    _, I = index.search(qv, 5)
    contexts = ["\n".join(texts[j] for j in row) for row in I]
    enc = enc.to("cpu")

    summary = {}
    for key in names.split(","):
        model_id, chat = SWEEP[key]
        try:
            llm = LLM(model=model_id, dtype="half", max_model_len=8192, gpu_memory_utilization=0.85,
                      enforce_eager=True)
            tok = llm.get_tokenizer()
            for prompt_kind in ("baseline", "template"):
                prompts = []
                for q, c in zip(dev["question"].to_list(), contexts):
                    if prompt_kind == "baseline":
                        user = (f"Bạn là một trợ lý thông minh. Trả lời câu hỏi dựa trên ngữ cảnh sau.\n"
                                f"Nếu không đủ thông tin, hãy nói rõ điều đó, không được tự bịa đặt.\n\n"
                                f"Ngữ cảnh:\n{c}\n\nCâu hỏi: {q}\nTrả lời ngắn gọn và chính xác nhất có thể:\n")
                        msgs = [{"role": "user", "content": user}]
                    else:
                        msgs = [{"role": "system", "content": TEMPLATE_SYS},
                                {"role": "user", "content": f"Ngữ cảnh:\n{c}\n\nCâu hỏi: {q}"}]
                    if chat:
                        kw = {"enable_thinking": False} if "qwen3" in key else {}
                        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))
                    else:
                        prompts.append(msgs[-1]["content"] if prompt_kind == "baseline"
                                       else TEMPLATE_SYS + "\n\n" + msgs[-1]["content"] + "\nTrả lời:\n")
                for max_tok in (256, 768):
                    t = time.time()
                    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_tok))
                    rows = [{"qid": r["qid"], "answer": r["answer"], "pred": o.outputs[0].text}
                            for r, o in zip(dev.iter_rows(named=True), outs)]
                    enc_gpu = enc
                    scored, means = _score(rows, lambda s: enc_gpu.encode(s, normalize_embeddings=True))
                    tag = f"{key}__{prompt_kind}__{max_tok}"
                    _save(job, f"{tag}.parquet", pl.DataFrame(scored))
                    summary[tag] = {**means, "seconds": time.time() - t}
                    _save(job, "summary.json", summary)
            del llm
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            summary[key] = {"error": repr(e)[:500]}
            _save(job, "summary.json", summary)
    _save(job, "DONE.json", {"ok": True})


# ---------- J4 training throughput probe ----------

@app.function(image=train_image, gpu="L4", volumes=VOLS, timeout=2 * 3600)
def trainprobe(models_csv: str = "Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B", minutes: float = 10, seq: int = 1024):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # law text only, skipping any corpus file whose name matches a test law id (no test knowledge in weights)
    test_laws = {x.replace("/", "_") for x in _qa("test")["law_id"].to_list()}
    files = [f for f in _corpus() if f.stem not in test_laws]
    report = {"files_used": len(files)}
    for mid in models_csv.split(","):
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                                                 task_type="CAUSAL_LM"))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        ids = []
        for f in files[:3000]:
            ids.extend(tok(f.read_text(encoding="utf-8", errors="ignore"))["input_ids"])
            if len(ids) > 5_000_000:
                break
        chars = sum(len(f.read_text(encoding="utf-8", errors="ignore")) for f in files[:300])
        toks = len(tok("".join(f.read_text(encoding="utf-8", errors="ignore") for f in files[:300]))["input_ids"])
        model.train()
        for bs in (8, 4, 2):
            try:
                steps, tokens, t0, pos = 0, 0, time.time(), 0
                while time.time() - t0 < minutes * 60:
                    batch = torch.tensor([ids[pos + i * seq: pos + (i + 1) * seq] for i in range(bs)], device="cuda")
                    pos = (pos + bs * seq) % (len(ids) - bs * seq - 1)
                    loss = model(input_ids=batch, labels=batch).loss
                    loss.backward(); opt.step(); opt.zero_grad()
                    steps += 1; tokens += bs * seq
                dt = time.time() - t0
                report[mid] = {"batch": bs, "seq": seq, "steps": steps, "tokens_per_s": tokens / dt,
                               "last_loss": float(loss), "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                               "chars_per_token": chars / toks}
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                report[f"{mid}__oom_bs{bs}"] = True
        _save("trainprobe", "report.json", report)
        del model, opt
        torch.cuda.empty_cache()
    # corpus size estimate in tokens
    total_chars = sum(f.stat().st_size for f in _corpus())
    report["corpus_bytes"] = total_chars
    _save("trainprobe", "report.json", report)
    _save("trainprobe", "DONE.json", {"ok": True})
