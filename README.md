# Vietnamese Law RAG — a sub-4B legal question-answering system

HCMUT *Large Language Models* course assignment: improve a naive RAG baseline for Vietnamese legal QA with an answering model **under 4B parameters**.
The final system (Qwen3.5-2B, supervised + two DPO rounds, dense retrieval + cross-encoder rerank over an extended legal corpus) reaches
**ROUGE-L 0.643 on the 200-question test split with one generation per question (0.652 with MBR-8)**, against 0.171 for the course baseline.

| | Cosine | Jaccard | Token overlap | BLEU-1 | ROUGE-L |
|---|---|---|---|---|---|
| Course baseline (test) | .602 | .164 | .191 | .138 | .171 |
| **Final, greedy** (test) | **.916** | **.673** | **.734** | **.582** | **.643** |
| Final + MBR-8 (test) | .918 | .681 | .728 | .587 | .652 |

* **Report** (scientific write-up, all experiments grouped and analysed): [`docs/report/report.pdf`](docs/report/report.pdf) (LaTeX source: `docs/report/paper/main.tex`).
* **Demo** transcript on 10 questions through the running service: [`docs/report/demo.md`](docs/report/demo.md).
* Chronological experiment log: [`docs/CHANGELOG.md`](docs/CHANGELOG.md); compute log: [`docs/ops/compute-log.md`](docs/ops/compute-log.md).

## 1. Quick start (no GPU, 1 minute)

```bash
uv sync --extra models --extra dev          # Python >= 3.11 (uv: https://docs.astral.sh/uv/)
make test                                   # unit tests (no data, no models)
make smoke                                  # whole pipeline on 5 questions: BM25 + a fake LLM
```

## 2. Data

Inputs are not in git (`data/` is ignored). Put the course files in `data/raw/`:

| file | what | source |
|---|---|---|
| `Dataset.zip` | 16,441 law texts (`export_1/*.txt`) | course Drive folder "RAG Assignment" |
| `res.csv` | the 2,745 question/reference rows (RES sheet) | same |

```bash
make data           # unzip the corpus, clean the QA sheet, write the fixed seeded split (test 200 / dev 300 / train 2,245) -> data/processed/
make vbpl-cited     # extra corpus: central-government laws since 2000 + every law the QA answers cite (HF dataset th1nhng0/vietnamese-legal-documents)
make web            # cited laws still missing, from public portal sitemaps -> data/processed/web   (polite crawl, rate-limited)
```
The retrieval store may contain full laws; **no model is ever trained on anything tied to the test split**: `src/corpus/decontam.py` drops training pairs that cite the same
(law, Điều) as a test answer or share a rare 13-gram with one (AGENTS.md, rule 3). Splits: `test` is for reporting, `dev` for choosing settings.

## 3. Reproducing the results

Everything is driven by a YAML experiment (`configs/exp/<id>.yaml`, only the differences from `configs/base.yaml`), a backend (`configs/backends/*.yaml`: where the LLM / embedder / reranker endpoints are) and `--set key=value` overrides.
A run writes `data/runs/<exp>__<hash>/{config.yaml, predictions.jsonl, metrics.json}` and a row in `data/runs/results.tsv`; same configuration = same directory (runs resume).
Embedding, reranking and the LLM are **inference endpoints** (vLLM, OpenAI-compatible); start them as in [`docs/ops/l40s.md`](docs/ops/l40s.md) (`make box ARGS="serve <model> --gpu 0 [--kind embed|rerank]"`) or any compatible server and set the URLs in the backend file.

```bash
# 3.1 course baseline (dev; add --set split=test for the report numbers)
make run EXP=configs/exp/B0_baseline.yaml
# 3.2 retrieval ablations: A2a ... A2e (cleaning, per-Điều chunks, extra corpora, de-duplication)
make run EXP=configs/exp/A2e_dedup.yaml
python -m cli retrieval data/runs/<run>          # doc@k / Điều@k / MRR   (PYTHONPATH=src)
# 3.3 supervised fine-tuning data + training (decontaminated; recipe "v3b": k in [3,10], gold injection 0.7, 0-3 shots, 20% refusal rows)
PYTHONPATH=src python -m cli sft-data configs/exp/A3_Q3.yaml --name q35_v3b --k-range 3,10 --oracle 0.7 --shots 0,1,2,3 --refuse 0.2
python src/train/sft_lora.py --data data/train/q35_v3b --base Qwen/Qwen3.5-2B --out data/models/q35_v3b --epochs 3 --max-length 12288 --batch 1 --accum 16
# 3.4 evaluate a model (serve it with vLLM first; shots=2 must be explicit)
make run EXP=configs/exp/A5_v3b.yaml ARGS="--set generate.model=data/models/q35_v3b --set generate.shots=2"
make run EXP=configs/exp/A6_mbrv3b.yaml ARGS="--set generate.model=data/models/q35_v3b --set generate.shots=2"   # MBR-8
# 3.5 on-policy DPO (round 1; round 2 repeats it from the round-1 model)
python scripts/pref_sample.py --data data/train/q35_v3b --out data/train/pref_v3b/cands.jsonl --model data/models/q35_v3b --base-url http://localhost:8001/v1
PYTHONPATH=src python scripts/build_pairs.py data/train/pref_v3b/cands.jsonl data/train/pref_v3b/pairs.jsonl --min-margin 0.1
python src/train/dpo.py --pairs data/train/pref_v3b/pairs.jsonl --base data/models/q35_v3b --out data/models/dpo_a --lr 2e-5 --beta 0.1
#   round 2: pref_sample with --model data/models/dpo_a (same train prompts) -> build_pairs -> dpo.py --base data/models/dpo_a --out data/models/dpo_a2
# 3.6 significance: paired bootstrap between two runs
make compare A=data/runs/<run_a> B=data/runs/<run_b> METRIC=rouge_l
```
Training scripts need the `train` extra (`uv sync --extra train`, or a separate environment with torch, TRL and PEFT); on the L40S box they run in a dedicated environment. Typical cost on one L40S:
SFT ≈ 1.3 h, one DPO round ≈ 35 min (+ 25 min sampling), one 300-question dev evaluation ≈ 5 min greedy / 12 min MBR-8. Weights are not in git (≈4.6 GB each); rebuild them with 3.3 + 3.5, or ask the authors.
GRPO (`src/train/grpo.py`), weight merging (`src/train/merge.py`), reranker / embedder fine-tuning and the ReAct agent are documented in the report and the notes under `docs/notes/`.

## 4. Running the QA service (container)

One question in, answer + sources + timings out (`POST /ask`); LLM, embedder and reranker are separate vLLM endpoints (compose file in `docker/`).
Full instructions, variables and measured latency (median 2.6 s end to end on one L40S): [`docs/ops/docker.md`](docs/ops/docker.md).

```bash
GPU=0 HF_CACHE=$HOME/.cache/huggingface docker compose -p lawrag-qa -f docker/compose.yml up -d     # needs data/models/dpo_a2, the index and data/serve/shots.json
curl -s -X POST localhost:8080/ask -H 'Content-Type: application/json' -d '{"question":"Mức phạt đối với hành vi không đội mũ bảo hiểm khi đi xe máy là bao nhiêu?"}'
python scripts/demo.py docs/report/demo_questions.json > docs/report/demo.md                          # the demo transcript (dev only)
```

## 5. Repository map

```
src/        corpus (data, decontamination) · index · retrieve · generate (prompt, generator, ReAct) · train (SFT, DPO, GRPO, merge) · evaluate · serve (QA service) · ops (GPU box)
configs/    base.yaml · exp/<id>.yaml · backends/ · queues/ (job queues for the GPU box)
scripts/    DPO sampling / pairs, analyses (MBR oracle, cascade, citation), demo, API evaluation
tests/      unit/ (no data, no models) · smoke/ (pipeline with BM25 + fake LLM)     docker/  compose + API image
docs/       report/ (paper, demo) · CHANGELOG · ops/ · notes/ · ADR/ · reference/ (assignment brief)
```
`AGENTS.md` is the entry point for coding agents (rules, layout, conventions).

## 6. Notes and limits

* Metrics are an exact port of the baseline notebook (`src/evaluate/metrics.py`); they measure similarity to template-style references, not legal correctness. Answers are not legal advice.
* The service refuses questions with no support in the corpus (best reranker score below `MIN_RERANK_SCORE`, default 0.001).
* Greedy decoding with vLLM is not bit-deterministic (±0.003 ROUGE-L between identical runs).
