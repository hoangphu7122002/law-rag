# CHANGELOG

## Step 1 — Read the assignment, scaffold the repo (2026-10-31)

- **Did:** read RAG_Assignment.pdf, Baseline.ipynb, Dataset.zip (16,441 law files), RES sheet (2,745 QA pairs);
  wrote the brief, a line-for-line port of the baseline metrics with tests, and the repo layout.
- **Next:** research directions (hybrid retrieval, structure-aware chunking, Graph RAG, LoRA on a < 4B model),
  then the baseline re-run on the fixed test split.

## Step 2 — `make data` and the first data finding (2026-10-31)

- **Did:** `make data` → 16,440 corpus files, 2,745 QA pairs, split test/dev/train 200/100/2,445 (seed 20261031).
- **Finding:** the QA set mostly asks about laws that are **not in the corpus**. 772 rows have no law id
  ("Không trích xuất được"); of the 1,973 with an id (314 distinct, mostly 2024–2025 banking circulars such as
  52/2024/NĐ-CP, 17/2024/TT-NHNN), only 8 ids / 51 rows match a corpus file name, and the top 5 ids appear in no
  corpus file text. Retrieval over the corpus alone cannot find the gold article for most questions.

## Step 3 — Research directions and the train-overlap finding (2026-10-31)

- **Finding:** 93 % of test questions with a law id cite a law also present in `train`, 57 % the same (law, Điều).
- **Research:** ranked directions in `docs/notes/01-research-directions.md`: LoRA/RAFT SFT on `train`, train-QA
  memory, hybrid + Điều chunking + reranker, a cheap regex law graph as the Graph RAG part. Default generator
  Qwen2.5-3B-Instruct (3.09B); Qwen3-4B (4.0B) is not under 4B.

## Step 4 — Roadmap settled (2026-10-31)

- **Decided:** due 2026-11-29; compute = Modal 1× L4 only. Roadmap in `docs/plan/roadmap.md`: core items, if-time items
  (DPO, GRPO, extra layers, LightRAG, …) and skipped items with reasons (verl multi-GPU, MS GraphRAG no delete,
  Graphiti/Mem0 need strong JSON LLMs, xRAG too little data, web lookup leaks answers).

## Step 5 — Roadmap widened to 90 USD, multi-L4 allowed (2026-10-31)

- **Decided:** 3 Modal accounts (~90 USD), multi-L4 allowed; almost every idea is planned. Risky ones run with guards
  (offline large extractor, anti-copy reward, verl timebox). Skipped only: web lookup at answer time, crawling Q&A
  sites (answer leakage), xRAG, Self-RAG, answering with a model ≥ 4B.

## Step 6 — Continued pretraining on a large crawled legal corpus (2026-10-31)

- **Decided:** crawl a large legal pretraining corpus (D10, all official texts, target ≥ 300M tokens) and legal Q&A pages
  (D11, guarded by 13-gram decontamination vs test/dev, own data version). Continued pretraining P-1: LoRA-CPT,
  extra layers, full CPT of a ≤ 1.7B model on 2–4 L4, then SFT on top.

## Step 7 — Rule: no test knowledge in any trained weights (2026-10-31)

- **Decided:** training data (pretraining, SFT/RAFT/DPO/GRPO, embedding fine-tune) excludes test/dev QA, train pairs
  citing the same (law, Điều) as test/dev, anything sharing a non-boilerplate 13-gram with test/dev answers, and the
  crawled text of test/dev-cited articles. The RAG store may hold everything except the test/dev QA pairs.
- **Measured on train (test only):** same (law, Điều) removes 16 % (keeps 2,050 / 2,445); rare 13-gram (in ≤ 3 answers)
  removes 20 % (keeps 1,966); dropping whole laws would keep only 57 % (too strict, not used). Dev filter still to measure.
- **Amended:** the knowledge filter is against `test` only. `dev` pairs are held out while tuning, then the final
  model is retrained on train + dev (also filtered against test).

## Step 8 — Dev 300 and the tuning method (2026-10-31)

- **Decided:** split test/dev/train = 200/300/2,245 (dev grew from 100 to cut selection noise). Tuning = grid search for
  discrete knobs + an autoresearch-style agent loop on `dev` only; a change is kept only if its gain passes a paired
  bootstrap test on `dev`. `test` is run once, by hand, at the end.

## Step 9 — First GPU runs on expiring Modal credit (2026-10-31)

- **Ran:** 4 jobs in parallel on 3 accounts (`scripts/vendors/modal/app.py`), ~4.0 USD, until each account's
  credit ran out. Log: `docs/ops/compute-log.md`. Outputs: `results/runs/20261031_modal/`.
- **Baseline on dev (300, vLLM, answer-only):** Qwen1.5-1.8B + course prompt, 256 tokens: cosine 0.592, Jaccard 0.163,
  token overlap 0.184, BLEU-1 0.126, ROUGE-L 0.155. With 768 tokens: token overlap 0.231, BLEU-1 0.116, ROUGE-L 0.132.
- **Prompt echo confirmed:** the course notebook pipeline returns prompt + answer on 25/25 test rows. Scored as-is (B0)
  token overlap is 0.778 vs 0.314 answer-only (B1); report both and use B1 as the fair baseline. The HF fp32 pipeline
  takes ~118 s/question, so the full B0/B1 rerun moves to vLLM or fp16 with the saved FAISS index.
- **CPT cost:** LoRA r64, seq 1024, one L4: 1.5B 1,537 tok/s, 3B 954 tok/s. 428 MB corpus ≈ 114M tokens →
  ~21 h (1.5B) / ~33 h (3B) per epoch. Full-corpus CPT on 3B is too costly; use 1.5B or a filtered 20–30M-token subset.
- **Embeddings cached** on the account-1 volume: c2000 × {bge-m3, Vietnamese_Embedding}, 188,228 chunks.

## Step 10 — Clean rebuild on LlamaIndex, L40S box, model choice (2026-11-17)

- **Decided:** LlamaIndex for RAG parts, TRL + PEFT for training, vLLM for serving; class-based flat `src/`;
  YAML + pydantic config with diff-only experiments; resumable `data/runs/<exp>__<hash>/` + `data/runs/results.tsv` ledger;
  retrieval cached apart from generation. ADR 0002; research in `docs/notes/05-model-choice/`, `06-codebase-design/`.
- **Moved:** `config/experiments` → `configs/exp`, `scripts/vendors/modal/app.py` → `src/ops/modal_app.py`, `results/runs` → `data/runs`, `report`/`notebooks`/`GLOSSARY`/`CHANGELOG` → `docs/`,
  `src/cli/data.py` → `corpus.DatasetBuilder`. New: `DieuNodeParser`, `IndexStore`, `RetrieverFactory`, `Retrieval`,
  `Prompt`, `Generator`, `RAGPipeline`, `Experiment`, `Evaluator`, `PairedBootstrap`, `Run`, `Ledger`, `RemoteBox`.
- **Compute:** 2× L40S box (shared) reachable over Tailscale; `src/ops/box.py` (`RemoteBox`, `make box`) drives it (only `law_rag/` and `lawrag-*` tmux).
- **Model choice (verified param counts):** answer LLM candidates Qwen2.5-3B-Instruct 3.09B (default),
  Qwen3.5-2B 2.27B, Vi-Qwen2-3B-RAG 3.09B, Arcee-VyLinh 3.40B, Qwen3-1.7B 2.03B; Qwen3.5-4B 4.66B excluded.
  Embedding: Vietnamese_Embedding (568M, self-reported Zalo legal Acc@1 0.727) vs bge-m3; reranker Vietnamese_Reranker.
- **Tests:** 21 pass (unit + smoke: BM25 + fake LLM on 5 dev questions).
- **Top level grouped by purpose:** src (code, incl. `src/ops/` for where it runs) · tests · configs (what to run) · data (inputs + outputs) · docs (to read).
- **Next:** B0 on L40S (dev + test), then C3 Điều chunks, R1 embeddings, R3 hybrid, R4 rerank, M sweep.

## Step 11 — ops moved into src/ops (2026-11-17)

- **Why:** the bash scripts in `ops/l40s/` duplicated box settings and had fragile quoting; now one class shares the config.
- **Added:** `src/ops/box.py` `RemoteBox` (push, push-data, setup, serve, run, status, stop, pull, tunnel) ·
  `box:` block in `configs/backends/l40s.yaml` (`config.Box`; `box.env` copies datamate's vLLM fixes, caches under `law_rag/.cache/`) · `Ledger.merge` · `RemoteBox.watch` + `autosync on|off` (launchd, code pushed on every save) · `cli box` · `make box ARGS=...` · `docs/ops/l40s.md`.
- **Moved:** `ops/modal/app.py` → `src/ops/modal_app.py` (frozen, excluded from ruff). Removed `ops/`.
- **Top level now:** src · tests · configs · data · docs.

## Step 12 — Extra corpus from vbpl.vn, test-blind (2026-11-18)

- **Why:** only 51/2745 QA rows (dev 5/300) cite a law that is in the course corpus; the corpus is mostly local QĐ/NQ
  and old, the QA is 2024–2025 banking/finance (52/2024/NĐ-CP alone: 146 questions). The assignment does not forbid
  extra documents ("any other improvements supported by quantitative metrics").
- **Source:** HF `th1nhng0/vietnamese-legal-documents` (CC BY 4.0), scraped from vbpl.vn (Ministry of Justice portal):
  171k docs metadata + HTML full text + 1M relations (amends / replaces / repeals).
- **Rule (fixed before looking at dev/test, metadata only):** scope Trung ương/Toàn quốc, issued ≥ 2010, no English
  translations, skip numbers already in the course corpus. 18,218 selected → 15,803 files (575 MB) in `data/processed/vbpl/`.
  No Q&A / advice pages (they may hold reference answers). Test law ids never read.
- **Coverage (questions with a law id whose law is in the corpus):** train 43 → 1,421 / 1,613; dev 5 → 197 / 218.
- **Added:** `corpus/vbpl.py` `VbplImporter` · `make vbpl` · config `sources: [corpus] | [corpus, vbpl]` (in `index_key`).
- **Report both:** course corpus only, and course + vbpl (ablation).

## Step 13 — vLLM for embedding and rerank, datamate serving flags (2026-11-18)

- **Why:** the AITeamVN embedder (568M, 512 tokens) ran at 77 chunks/s in sentence-transformers (fp32), so one index
  of the course corpus took ~45 min.
- **Measured on L40S:** vLLM pooling server 449 chunks/s (5.8x), vectors equal to local (cosine 0.9999).
- **Added:** `embed.engine` / `rerank.engine: vllm | local` (default vllm; B0 stays local) · `backend.embed_url`,
  `rerank_url` · `retrieve/vllm_rerank.py` `VllmRerank` (`/rerank`) · `make box ARGS="serve <m> --kind llm|embed|rerank"`
  (ports 8000/8100/8200 + gpu, sessions `lawrag-<kind>-g<gpu>`, all three on one GPU via `gpu_share`) ·
  LLM flags as datamate: bf16, prefix caching, 64 sequences, 16k batched tokens · `generate.concurrency` 64.
- **Fixed:** `RemoteBox.shell` uses `bash -c`; the box's `.bash_logout` turned `exit 0` into exit 1 under `bash -l`.

## Step 14 — Corpus exploration (2026-11-18)

- **Added:** `docs/notebooks/01-corpus-eda.py` → `docs/report/corpus-eda.md` + `docs/report/figures/corpus-*.png`
  (per-doc stats also in `data/processed/corpus_stats.parquet`). QA law ids from train + dev only.
- **Found:** 16,439 docs, 63M words, median 2,224 words/doc; 65 % local authorities (UBND/HĐND), 0.4 % Quốc hội;
  median year 2015, only 1,115 docs from 2024+; 12,272 docs have Điều (84k Điều in all); `Nơi nhận` tail ~3k chars.
- **Found:** every file ends with relation blocks (`Van ban duoc can cu`, `thay the`, `sua doi bo sung`,
  `huong dan`, …): 191k edges, a ready-made document graph for roadmap phase 4. Relation blocks are currently
  embedded with the body (noise in chunks).
- **Found:** of 2,545 train + dev questions, the cited law is in the corpus for 48; 127 more are only named in a
  relation block. 926 of the cited laws are from 2024–2025.

## Step 15 — Public legal datasets for training (2026-11-18)

- **Why:** retriever fine-tuning (roadmap A7) and generator SFT need (question, article) pairs beyond our 2,245 train rows.
- **Added:** `corpus/extra.py` `ExtraData` · `make extra [ARGS="zalo"]` → `data/raw/extra/<name>/` (on the box, not in git).

| name | HF repo | content | size |
|---|---|---|---|
| zalo | minhnguyent546/zalo-ai-legal-text-retrieval-2021 | Zalo AI 2021: 3,196 questions → article, 61k articles | 33 MB |
| bkai | tmnam20/BKAI-Legal-Retrieval | 119k questions → passages, 262k-passage corpus | 153 MB (750 MB unzipped) |
| legal_qa | thangvip/vietnamese-legal-qa | 29k LLM-generated QA over 9.7k articles of 2024+ laws | 21 MB |

- **Skipped:** `duyet/vietnamese-legal-instruct` (1.5 GB synthetic, other answer styles), `truro7/vn-law-*` (ids only),
  `nguyenlab/ALQAC` (gated, needs an account accept).
- **Overlap with RES questions (counts only):** exact match 0 for all. Near-duplicates (≥ 50 % of a RES question's
  word 6-grams found): legal_qa train 52 / dev 6 / test 7; bkai 36 / 5 / 4; zalo 0 / 1 / 0.
  → these must be removed by the decontamination filter (`corpus/decontam.py`, still to write) before any training use.
- **Licenses:** none stated on the three cards; use for the course assignment only and cite the repos.

## Step 16 — LawCleaner, corpus package split, A2 configs (2026-11-18)

- **Decision:** the lecturer allows external data → course + vbpl is the main setting (course-only kept as ablation).
- **Layout:** `corpus/sources/` (CorpusReader, VbplImporter, ExtraData) · `corpus/ingest/` (LlamaIndex transforms:
  LawCleaner, DieuNodeParser, patterns).
- **Added:** `LawCleaner(TransformComponent)`, enabled by `chunk.clean: true`; `index.Chunker` now runs a LlamaIndex
  `IngestionPipeline([LawCleaner?, parser])`. Removes: header (issuer, quốc hiệu, motto, Số, date), "Nơi nhận" +
  signature blocks (attached rules / appendices after them kept), blank-form lines, Word hard wraps, the course tail
  blocks (THÔNG TIN THÊM, MỤC LỤC, VĂN BẢN LIÊN QUAN). The info block and the vbpl manifest become metadata
  (number, type, field, issuer, issued, effective, status, level, year); relation edges come out of
  `LawCleaner.parse` for the graph step.
- **Title in context:** metadata `title` is the only key shown to embedding and LLM (node template
  `{title}\n{content}`); retrieval / rerank read `get_content(MetadataMode.LLM / EMBED)`. Old indexes unchanged
  (all their metadata is hidden); `index_key` ignores `clean` when false, so B0 / A1 caches still hit.
- **Measured:** course corpus 329M → 255M chars (77 %), 0 empty docs, title for 16,439/16,439; vbpl 429M → 414M.
- **Fixed:** `DieuNodeParser.unwrap` joined whole vbpl files into one line (no blank lines) → only unwraps Word exports.
- **Experiments:** `A2a_clean` (clean, recursive), `A2b_dieu` (clean, per Điều), `A2c_vbpl` (+ vbpl), all A1 stack, dev.
- **Results (dev 300, vs A1 .6623/.2281/.2153/.1344/.1670, law_hit .013):** A2a .6657/.2286/.2275/.1455/.1695 (.013);
  A2b .6655/.2278/.2242/.1411/.1714 (.017); **A2c .7935/.4665/.5074/.3745/.4148, law_hit .590**. Cleaning alone
  is a small gain; the corpus coverage is the bottleneck (A1 retrieved the right law for 1 % of questions).
  vbpl selection is test-blind (metadata rule only), so A2c is the new reference config.
- **Test peek (user decision, 2026-11-18):** the test `law_id` column (not answers) was read to see corpus gaps.
  Test 200: course 3, vbpl 121, vbpl excluded by year 4 (1999–2007 QĐ, incl. 571/2002/QĐ-NHNN), not extractable 58,
  not in vbpl 14 (internal NHNN / KBNN / BHTG / BHXH decisions and notices, 72/2025/QH15). Any corpus change after
  this point is test-informed; the report must say so. Training data stays test-free.

## Step 17 — Milestones M0/M1 on test, M2 sources (2026-11-18)

- **Milestones:** M0 = course data only (A2a/A2b), M1 = + vbpl test-blind rule (A2c), M2 = + cited vbpl
  (`vbpl_cited`, >= 2000 + every cited doc) + luatvietnam.vn crawl (`web`). Each milestone is its own folder.
- **Test 200 (B0 / A1 / A2a / A2b / A2c), cos-jac-tok-bleu1-rougeL:** B0 .6019/.1639/.1914/.1382/.1709;
  A1 .6708/.2263/.2106/.1256/.1713; A2a .6639/.2330/.2252/.1377/.1755; A2b .6675/.2315/.2227/.1385/.1758;
  **A2c .8107/.4918/.5406/.4047/.4369 (law_hit .545)**.
- **New:** `corpus/sources/citations.py` (documents cited in QA answers, resolved course / vbpl / web: 18 / 425 / 150),
  `corpus/sources/luatvietnam.py` (sitemap lookup, robots.txt-compliant, 1 req/s; 105 / 150 matched).
- **Roadmap:** status + ordered next steps (A2d → retrieval eval → A4 → A3 → decontam → distillation → SFT/GRPO),
  distillation block (KD0–KD4, teacher offline), Condenser / ensemble moved to low priority with reasons,
  Qwen3.5 sizes (2B = 2.27B allowed; 4B = 4.66B not).
- **Research:** `docs/notes/08-weight-space-techniques/` (inference edits, knowledge injection, merging + report);
  roadmap block W0–W5 (decoding hygiene, copy bias, WiSE-FT / soups, CAD, steering, knowledge modules, chat vector).
- **Order rule:** Tier 1 basics (data milestones, retrieval, prompt / decoding, LLM sweep, decontam, SFT / RAFT,
  embedding fine-tune) before Tier 2 advanced (distillation, weight-space W1–W5, GRPO, graph, ensemble, Condenser).
- **Ablation (test 200), course models (sbert + Qwen1.5-1.8B):** B0 raw .6019/.1639/.1914/.1382/.1709 · B1a cleaned
  .5735/.1469/.1868/.1356/.1633 (cleaning alone does not help sbert: B0 lower-cases + flattens) · B1c + vbpl
  .6292/.1871/.2222/.1667/.1908, law_hit .345. Same M1 data with the AITeamVN stack (A2c): .8107/.4918/.5406/.4047/.4369,
  law_hit .545 → data and models compound; the strong stack gains far more from coverage.
- **sbert on vLLM:** `engine: vllm` matches local sentence-transformers (cos 0.9999, same norms), 2.7× faster.
- **M2 (A2d, course + vbpl_cited + web, 726k nodes):** dev .8015/.4682/.5278/.3892/.4210 (law_hit .577 vs A2c .590);
  test .8116/.5009/.5561/.4130/.4440 (law_hit .555 vs .545). Small gain (+.007–.02 tok / bleu1); more coverage also
  adds distractors → retrieval ranking, not coverage, is now the bottleneck (next: retrieval eval + A4).
- **Dedup audit (M2):** 110 exact content duplicates (0.24 %), 12 web docs already in vbpl_cited, ~40 same-number
  other-year web docs, web pages keep signature lists and an HTML fragment. Fix pending (MinHash dedup transform).

## Step 18 — Gold Điều set, retrieval eval, corpus dedup (2026-11-18)

- **`corpus/keys.py`:** one key per document (`law_key`, `stem_key`): NFC, upper, `_` = `/`, `-TTg` = `-TT`, drops
  web / vbpl `__<id>` and course `_1` / `_2` copy suffixes. `Evaluator.law_hit` uses it (web docs never matched before).
- **`corpus/gold.py`:** (document, Điều) pairs parsed from every answer: each "Điều N" goes to the next cited
  document within 200 chars, else the question's law_id; laws by name resolved via vbpl titles (`Citations.resolve`,
  new `Citations.spans`). 2,607 / 2,745 questions with a document, 2,432 with a Điều. Scoring only, never training.
- **`evaluate/retrieval_eval.py` (`cli retrieval <run | cache.jsonl>`):** doc@k / Điều@k / MRR with LlamaIndex
  `HitRate` / `MRR`, ids = law keys and "key#Điều".

| run | split | doc@1 | doc@5 | Điều@1 | Điều@5 | doc MRR |
|---|---|---|---|---|---|---|
| B1c (sbert, M1) | dev | .224 | .459 | .102 | .277 | .316 |
| A2b (M0) | test | .016 | .021 | .011 | .017 | .018 |
| A2c (M1) | dev / test | .609 / .643 | .879 / .871 | .576 / .616 | .879 / .848 | .722 / .733 |
| A2d (M2) | dev / test | .644 / .648 | .932 / .922 | .591 / .610 | .905 / .898 | .765 / .755 |

  → the gold Điều is in the top 5 ~90 % of the time on M2; the answer step (A3) now limits the score more than
  retrieval. M0 retrieval is near zero: the course corpus lacks most cited documents.
- **Corpus (LlamaIndex transforms):** `corpus/ingest/dedup.py` `DocDedup` (exact text hash + same-number 5-gram
  Jaccard >= 0.6; first source wins), `chunk.dedup` flag (old index keys unchanged); `DieuNodeParser` adds node
  metadata article / heading / chapter (hidden from embed + LLM text); luatvietnam body no longer ends in a
  `<div class="` fragment (160 web docs after rebuild, 6 wrong-number stale files removed). A2e = A2d + these.
- **A2e (M2 + DocDedup, 186 docs dropped: 138 exact, 48 near → 724,215 nodes):** dev .8026/.4721/.5325/.3916/.4229,
  test .8119/.4996/.5551/.4115/.4433 — same as A2d within noise; retrieval identical (test Điều@5 .898).
  law_hit .633 / .620 (vs .577 / .555) comes from the `law_hit` key fix, not dedup.
- **k sweep (A2d, `cli run --retrieval-only`):** test Điều@5 / @10 / @20 = .898 / .944 / .966, doc .922 / .959 / .969;
  @1 stays .61 / .65 → try k = 10 in A3; reranker ordering (A4) is the @1 gap.
- **Box safety (ops/box.py):** the first parallel attempt exhausted the 62 GB box (hard reboot; kernel 6.8.0-149 then
  lacked the NVIDIA module, fixed by the owner). Now every launch passes a guard (RAM >= 24G, disk >= 20G, one run at
  a time), runs under `systemd-run --scope -p MemoryMax=32G`, and `cli box queue <file>` chains jobs on the box.
  DocDedup keeps shingles only for repeated numbers (RAM).

## Step 19 — A3 decoding: sampling, thinking on/off, MBR (2026-11-18)

- Research (`docs/notes/09-inference-configs/`): official sampling per reader; datamate's final inference used
  thinking ON (client-side `enable_thinking=True`, T 0.6 / top_p 0.95 / top_k 20, 5-sample vote).
- `Generate` gains `top_p, top_k, min_p, presence_penalty, repetition_penalty, thinking, think_tokens, mbr`;
  `run_key` ignores them at their defaults, so old run keys are unchanged.
- `Generator`: vLLM-only params via `extra_body` (+ `chat_template_kwargs.enable_thinking`); thinking path renders
  the chat template client-side, completes the thought up to `think_tokens` (stop `</think>`), forces `</think>`,
  then completes the answer; `mbr` samples one answer per temperature and keeps the most central by ROUGE-L.
- `RemoteBox.queue`: a `serve <model> --gpu N` line swaps our LLM server between runs (queue stops if it fails).
  LLM server gets `--max-model-len 32768`.
- Configs `A3_G1, A3_Q0–Q3, A3_R0–R1` (retrieval = A2e); `configs/queues/a3.txt` runs them on dev.

## Step 20 — SFT v0, decontamination, reranker fine-tune, KD0 (2026-11-19)

- `corpus/decontam.py` (AGENTS.md rule 3): drops train/dev pairs that cite a test (law, Điều) or share a non-boilerplate
  13-gram with a test answer; train 2245 → 1452, dev 300 → 188.
- `train/sft_data.py` + `train/sft_lora.py` + `cli sft-data`: LoRA SFT (r32, lr 1e-4, completion-only loss, val loss
  every 30 steps, best checkpoint, merge). Prompt rendered with the inference chat template (thinking off).
  Bugs: base weight shard / index copied into the merged dir (vLLM would serve base) → `complete_config`
  copies only configs, `clean_merged` repairs old dirs; `warmup_ratio` → `warmup_steps=0.05` (transformers 5).
- `train/rerank_data.py` + `train/rerank_ft.py` + `cli rerank-data`: course dense top-50 (TopK-PercPos filter) + Zalo
  2021, decontaminated against test; BCE CrossEncoder, best dev NDCG@10. Bug: `.venv` lacked sentencepiece.
- `Generate.teacher` flag (params_b ≥ 4 only with `teacher: true`; KD0 measurement, never reported).
- `box queue`: `sh`, `wait`, `do` lines, `--parallel`, `serve --share`, `MAX_RUNS=2`. Caveats: `do` runs inline (does not
  wait for runs; add a `wait` first); a run with errors is cached, delete `data/runs/<run>` and its results row to redo.
- **Results (dev, cos/jac/tok/bleu1/rougeL):** A2e Vi-Qwen2 raw .8026/.4721/.5325/.3916/.4229 → SFT Qwen3.5-2B
  .8982/.6454/.7577/.5549/.6026 (paired bootstrap vs A2e: p_b_not_better 0.0 on rougeL, bleu1, cosine); SFT Vi-Qwen2
  .8965/.6376/.7424/.5531/.5986; k=10 and thinking on not better; reranker rr_v0 lifts retrieval (art@1 .587 → .712) but
  not generation. KD0 9B teacher rougeL .411 / .429 (< SFT 2B).
- Caveat: 151/188 dev answers share a 13-gram with train answers (75 > 50 %), mostly answer templates → part of the
  dev gain is style; judge on test only once, for the final config.

## Step 21 — SFT v2/v3/v3b data, few-shot, refusal rows, test-informed check (2026-11-19)

- `sft-data` flags `--k-range`, `--oracle` (gold Điều injected when top-k misses it), `--shots` (nearest train Q/A pairs,
  `generate/shots.py`, `Generate.shots`), `--refuse` (no-answer rows: context of an unrelated question of another law →
  refusal text; the answerable original is kept, datamate "contrast" rule). Datamate datagen read: `docs/notes/11-datamate-datagen.md`.
- Always retrained from base Qwen3.5-2B. Dev rougeL (cos/jac/tok/bleu1 in compute-log): v0 .6026; v2 (dynamic k + oracle) .5938;
  v3b (shots 0–3, 20 % refusal, k 3..10) shots2 **.6325**, shots0 .5937, k10 .6122; v3 shots2 .6266 (k10 .6221, shots0 .5956);
  hp A/B (alpha64, dropout0, lr2e-4, 2 ep, A5_h) .5988; v3b + fine-tuned reranker rr_v0 .6253 (no gain).
- **Test-informed (not tuned on, 200 q):** v3b shots2 k5 .9053/.6562/.7347/.5380/.6073 vs v0 k5 .9050/.6583/.7541/.5631/.6175
  → the dev gain of v3b did NOT transfer (dev/train template overlap); v0 stays the safer pick (paired bootstrap test rougeL v0−v3b +.010, CI [−.011, +.032], not significant). Both errors=0.
- Queue gotchas: tmux session names are truncated → distinct exp stems + `wait ! tmux ls | grep -q lawrag-run-<stem>`;
  `serve` right after a kill can fail the free-GPU check → `sleep 40`.
- Error analysis: `docs/notes/12-error-analysis.md`. Next: length control, Haiku-generated QA (paced), GRPO fallback.
- **Full test pass (test-informed, 200 q, all errors=0, reported only; cos/jac/tok/bleu1/rougeL):**
  v0 k5 .9050/.6583/.7541/.5631/.6175 · v0 k10 .9065/.6571/.7665/.5689/.6203 · v2 k5 .9050/.6546/.7741/.5668/.6076 · v2 k10 .9120/.6639/.7826/.5733/.6160 ·
  v3 s0 k5 .9102/.6661/.7418/.5710/**.6212** · v3 s2 k5 .9107/.6533/.7382/.5642/.6086 · v3 s2 k10 .9060/.6497/.7427/.5687/.6122 ·
  v3b s0 k5 .9041/.6607/.7488/.5613/.6069 · v3b s2 k5 .9053/.6562/.7347/.5380/.6073 · v3b s3 k5 .9067/.6645/.7413/.5455/.6100 ·
  v3b s2 k10 .9072/.6534/.7488/.5406/.6072 · v3b s2 + rr_v0 .9045/.6575/.7456/.5547/.6101 · hp (A5_h) k5 .9056/.6540/.7787/.5663/.6126.
  All variants fall in rougeL .607–.621 (spread ≈ bootstrap noise ±.02): the dev gains of shots/refusal rows/dynamic k do not
  separate on test; extra data/recipes of this kind are not the lever → RL fallback next.

## Step 22 — new test-like dev, MBR on SFT v0, GRPO first pilot (2026-11-19)

- `corpus/devsets.py`: plain dev shares a Điều with kept train for 134/300 (45 %) and a non-boilerplate 13-gram for 216/300 (72 %);
  test shares 0 Điều and 63/200 (32 %). New splits `dev_clean` (166, no shared Điều) and `dev_strict` (74, also no shared 13-gram),
  `data/processed/dev_sets.json`, `--set split=dev_clean|dev_strict` (reads QASet). v3b − v0 rougeL: dev300 +.030, dev_clean +.052 [+.024, +.079],
  dev_strict +.032 [−.008, +.072], test −.010: the shot/refusal gain shrinks with overlap but does not vanish on dev_clean.
- **MBR on SFT v0** (greedy + 7 samples T 0.7, ROUGE-L pick, A6_mbr): dev .6366 (v0 .6026, +.034 [+.020, +.048]), dev_clean +.041 [+.023, +.061],
  dev_strict +.042 [+.005, +.081], **test .6451 vs .6175, +.0276 [+.011, +.044]** — the first gain that holds on test (test-informed, pre-declared pair).
- GRPO pilot (c1 reward .6 rougeL+.2 tok+.2 cite − copy − two-sided length; c2 no length term; 50 steps × 4 prompts × 8 gens, lr 1e-5, beta 0, LoRA r32,
  from merged v0, HF generation ~90 s/step, 75–77 min each): dev .6101 / .6064 vs v0 .6026 (+.0075 [−.003, +.017] / +.004): not significant at 50 steps.
- Infra: `.rl_env` (torch 2.13, vLLM 0.30, TRL 1.15) deleted twice by `rsync --delete` (box push / a long-running `box watch` with the old excludes) →
  `.rl_env` now in `CODE_EXCLUDES`, watcher restarted. GRPO checkpoint needs the base's processor json files (copied from the SFT dir now).

## Step 23 — ReAct baseline harness, KG store (2026-11-19, code only; no GPU runs yet)

- `generate/react.py` + `Generate.agent / max_turns / agent_initial_k`: ReAct loop on the raw Qwen3.5-2B with native tool calls
  (`search_law`, `get_article`), datamate loop guards, 5 unit tests; `serve ... --tools` adds the vLLM tool-call flags; `configs/exp/A7_react.yaml`,
  queue `configs/queues/react_g0.txt` (runs after the GRPO queues). Notes: `docs/notes/13-react-baseline.md`, research in `docs/research/react-tools/`.
- Measured: gold Điều in top-5 for 239/264 dev and 159/177 test questions; only 2 / 3 of the misses have the gold Điều within ±1 of a retrieved one →
  no `neighbors` tool.
- `src/kg/` rule-built legal KG in SQLite from vbpl metadata + relationships (171,556 docs, 989,692 edges, `cli kg-build`), updatable
  (upsert / remove / as_of); notes `docs/notes/14-kg.md`. Retrieval stays dense + rerank (hybrid BM25 exists in the factory, never measured).
- Tests first from now on (memory `tdd-for-new-code`): KG store backfilled (21 tests: validity windows, family map of all 31 vbpl relation names, build from parquet,
  persistence, update/delete) — two real bugs found by RED tests (documents without a law number all matched the empty key; the same neighbour listed twice because vbpl
  records each relation on both documents). `KG.related(hops, per_hop, max_total)` + `render_related` + `kg_related` ReAct tool + `generate.agent_tools` built
  RED→GREEN (84 tests pass). Nothing of the KG has run on a GPU yet.
- Web search tool (TinyFish provider, cache, domain allow-list, test-answer leak filter) and `generate.agent_tools` / `generate.web` switches built test-first (98 tests);
  design + arms T0–T3 in `docs/notes/15-web-search.md`; draft queue `configs/queues/tools_g0.txt` (not queued). No external call made yet.
- Billing facts: TinyFish Search is free only on Hobby (30/min); Builder = 150/min at $0.005 / query. Spend cap (`generate.web.max_cost_usd`, persistent over runs), key from a 600 file
  (`generate.web.key_file`), rate limits in config; 108 tests. Arms T1 (KG) and T2/T3 (web, Builder, cap $6) queued on the raw Qwen3.5-2B after the ReAct baseline (queues 185128 / 185131).
- Optional answer gate `generate.check` (`generate/check.py`, test-first): `grounded_citation`; MBR drops failing candidates (fallback: keep all), ReAct nudges once. Reference answers fail it in 17 % of dev
  questions (retrieval misses), so it is a preference, not a veto; untested on a GPU run. 118 tests.

## Step 24 — codebase reorganisation, phase 1 (2026-11-19)

- Committed the 143 pending changes in six code groups (corpus, index/retrieve/evaluate, train, generate, kg, ops/cli) + configs + docs, then reorganised (c08e97a).
- Moved (git mv, no content change): `configs/queue_*.txt` → `configs/queues/*.txt` (43 files), `research_*` → `docs/research/*` (4 folders); references updated.
- `docs/reference/structure.md` rewritten from the real imports (kg, train, ReAct / web / gate, dev subsets, import-rule table); `AGENTS.md` layout + rule 9 (test first).
- Deliberately NOT moved yet: `src/train/` and `configs/exp/` (queued GRPO jobs on the box call `src/train/grpo.py`); no `generate/` → `agent/` split (all files < 300 lines today).
  Phase 2 after the GRPO and ReAct queues finish: optional `agent/` package (react, web, check), box-side helper scripts into `scripts/`, old queue files pruned.

## Step 25 — embedder benchmark on Modal (2026-11-19)
- Added `cli embed-bench` (src/evaluate/embed_bench.py, tests) + Modal job src/ops/modal_embed.py; results in docs/research/model-selection/embed_bench_results.md.
- Result: AITeamVN Vietnamese_Embedding stays best on the pool (Điều@1 .534, MRR .674); Qwen3-Emb-4B/8B .51 @1 but 9-12x slower; bge-m3 .523. Pool is AITeamVN's top-50 (biased); no switch.
- GRPO (Step 26 pending c3f/c4f): c1f dev greedy .6225 vs v0 .6026 (+.020, CI .0065..0338), c2f .6116 (+.009, ns), c1f MBR8 .6374 vs MBR8 v0 .6366 (no gain over MBR).

## Step 26 — embedder fine-tuning on Modal: negative / borderline result (2026-11-19)
- src/train/embed_ft.py (MNRL, LoRA/full/query-only/interpolation) + src/ops/modal_embed_ft.py + paired bootstrap in embed-bench; 15 variants on AITeamVN with 4,015 rr_v0 groups.
- lr 1e-4 hurts (Điều@1 .46-.49 vs .534). Conservative lr 2e-5 / scale 50: +.023 Điều@1 [+.004,+.045] but strict-subset lower bound 0 and no seed reproduction → not accepted; no re-index. Details: docs/research/model-selection/embed_ft_results.md.

## Step 27 — overnight 2026-11-19/20 (interim, updated by the monitor)
- RL dev greedy rougeL (v0 .6026; paired bootstrap): c1f .6225, c2f .6116 (ns), **c3f cosine .6343 (+.032 [.019,.045])**, c4f beta .01 .6274 (+.025), c5f cosine lr 2e-5 .6222 (+.020; clean/strict ns), c6f cosine+beta .01 .6242 (+.022). MBR8: c3f .6450, c4f .6376, c5f .6407 (v0 MBR8 .6366). All RL variants within ~.012 of each other = noise level; reward curves saturate (~.73); no LR sweep needed.
- **Compact tool prompt** (drop the duplicated per-tool prose list, keep the answer template): ReAct on SFT v0, dev_clean .5926 vs .5709 (full prompt), diff +.0217 [+.004, +.041]; dev_strict +.026 [+.003, +.056]. Now above plain RAG v0 greedy (.5828). ReAct data collection and SFT rebuilt with `generate.tool_prompt=compact`.
- **9B teacher gate FAILED**: raw Qwen3.5-9B on dev_clean: ReAct .436, plain RAG .398 (SFT v0: .571 / .583) -> a raw 9B teacher does not reproduce the reference style; on-policy distillation from it is dropped (a fine-tuned teacher would be needed).
- Other negatives: chunk context line (chapter/scope) no retrieval gain (paired CI spans 0); embedder fine-tune not accepted (Step 26); kNN prior only +1.5-3.3 pts recall on clean dev. ROUGE-L survey: no verified 0.8-0.9 for Vietnamese legal QA (docs/research/rougel-benchmarks/).
- **Retrieval is not the bottleneck** (oracle = gold Điều at rank 1, v3b + 2 shots, greedy, dev300): .6367 vs C1 .6325, +.004 [-.003,+.012] ns. Even perfect retrieval/ReAct-search buys <= ~.01; gains must come from generator/style (MBR, RL, SFT data). The earlier oracle MBR8 run (.6329) lacked model/shots overrides -> not comparable, ignore.
- **ReAct S1 SFT (816 trajectories from v0 compact, seed 1, train)**: trajectories had 0 tool calls (v0 compact never calls tools) -> S1 = plain SFT. react_s1 dev_clean .6011 vs v0 compact .5926 (+.0085 [-.019,+.033] ns), 0/166 trajectories use a tool. Given the oracle bound, S2 repair (DeepSeek V4 Flash / Haiku) for tool use is low value; not pursued unless the user insists.
- Weight merging: arith_v3b_c3f_a1 (2 shots) dev300 .6516 vs C1 greedy +.019 [+.006,+.034] but dev_clean +.009 [-.012,+.029], dev_strict +.009 [-.025,+.043] -> not accepted; a05 .6406; soup_rl3 zero-shot .6301 (< c3f .6343). c6f MBR8 .6391 (< c3f .6450). c3s2 seed repeat OOM at step 72/150 (GPU0 shared) -> dropped.
- Ops lessons: tmux run-session names truncate and collide across GPUs (same A7_react/A5_v3b name) -> 'duplicate session' skips runs; release race 'free<20000 refusing' after kill; react_sft must copy preprocessor files (fixed).
- **Provisional C2** (see docs/plan/candidates.md): task-arithmetic merge arith_v3b_c3f_a1 (v3b + (c3f - v0)) + 2 shots + MBR8 = .6668 vs C1 .6575; paired vs C1 dev_clean +.017 [+.001,+.035], dev_strict +.021 [+.0005,+.043], dev300 +.009 ns. a05 merge MBR8 .6642. Marginal (~10 variants compared) -> confirm with a seed repeat.
- **Invalid runs (shots=0 by mistake)**: A6_mbrv3b.yaml has no shots field; MBR runs without `--set generate.shots=2` (arith_a1 .6346, oracle .6329, T-mix .6197, T0.5 .6209) are not comparable to C1 and were redone with shots 2. Valid: MBR T0.5 .6428, T0.9 .6582 (~C1), MBR16 v3b .6586 (ns), rl_v3b_c3 greedy .6271 / MBR8 .6512 (< C1).
- Ops lessons: `sh` queue lines do not block (add `wait ! tmux ls | grep -q lawrag-run-<name>`); tmux run names collide across GPUs and with same-named runs; a serve right after a kill can be refused (<20G free); GPU0 is shared with other users (~6G) so GRPO with long prompts can OOM (use --max-prompt-tokens 3500).

## Step 28 — plan
1. Confirm C2 with a second MBR sampling pass (queued) and MBR16; if it holds, C2 replaces C1.
2. Final retrain on train+dev for v3b + c3f RL and merge (alpha 1), then ONE test run of the chosen candidate.
3. Latency per stage; report, user manual, demo. ReAct stays an optional extra (retrieval is not the bottleneck: oracle +.002 ns).
- MBR noise check: C1 second pass .6512 (first .6575); C2 three passes mean .6638 vs C1 mean .6544 (+.009, ~ sampling noise ±.006) -> C2 provisional, C1 stays the safe candidate.
