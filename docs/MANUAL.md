# law-rag: user manual

How to install, run and reproduce the submitted system: a Vietnamese legal question-answering RAG with a Qwen3.5-2B generator (fine-tuned with LoRA SFT, then two rounds of on-policy DPO). Every number in the report can be rebuilt with the commands below.

| Item | Where |
|---|---|
| Report | `report.pdf` (LaTeX source: `docs/report/paper/`) |
| Source code | this repository (`source/law-rag-src.zip` in the submission) |
| Dataset (small files) | `dataset/` in the submission |
| Dataset (corpus, training data) | https://huggingface.co/datasets/hoangphu7122002ai/law-rag-data |
| Model weights | Hugging Face Hub, table below |

| Model version (merged, bf16) | Hugging Face | Dev / test ROUGE-L (greedy) |
|---|---|---|
| **Reported system**: SFT v3b + 2 DPO rounds | https://huggingface.co/hoangphu7122002ai/law-rag-qwen3.5-2b-dpo | **.664 / .643** |
| SFT v3b (supervised stage of the reported system) | https://huggingface.co/hoangphu7122002ai/law-rag-qwen3.5-2b-sft | .633 / .607 |
| PiSSA SFT (best supervised model, report "Post-training from the PiSSA supervised model") | https://huggingface.co/hoangphu7122002ai/law-rag-qwen3.5-2b-pissa-sft | .641 / .631 |
| Fixed configuration on train+dev, SFT (report "Final configuration") | https://huggingface.co/hoangphu7122002ai/law-rag-qwen3.5-2b-trainval-sft | – / – |
| … + DPO round 1 | https://huggingface.co/hoangphu7122002ai/law-rag-qwen3.5-2b-trainval-dpo1 | – / – |
| … + DPO round 2 | https://huggingface.co/hoangphu7122002ai/law-rag-qwen3.5-2b-trainval-dpo2 | – / .640 |
| Scaling point, same recipe on Qwen3.5-4B (4.66B, over the limit: analysis only) | https://huggingface.co/hoangphu7122002ai/law-rag-qwen3.5-4b-dpo | .679 / .666 |
| Scaling point, same recipe on Qwen3.5-9B (9.65B, analysis only) | https://huggingface.co/hoangphu7122002ai/law-rag-qwen3.5-9b-dpo | .700 / .677 |

Main result (200 held-out test questions, ROUGE-L): course baseline .171, **final system .643 (greedy)**, .652 with MBR-8.

---

## 1. Requirements

| | Minimum | Used for the report |
|---|---|---|
| OS | Linux (x86_64). macOS works for tests and the smoke run only | Ubuntu 22.04 |
| GPU | 1 × NVIDIA GPU with 24 GB to run the system; 40 GB+ to train | 2 × L40S 46 GB |
| RAM / disk | 32 GB RAM, 60 GB free disk (the index alone is 12 GB) | 62 GB RAM |
| Software | Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), vLLM 0.30 (separate environment), `hf` CLI | same |

```sh
unzip law-rag-src.zip && cd law-rag            # or: git clone https://github.com/hoangphu7122002/law-rag.git
uv sync --extra models --extra dev             # pipeline + tests
uv sync --extra train                          # only to retrain (TRL, PEFT)
python -m venv ~/vllm_env && ~/vllm_env/bin/pip install vllm==0.30.0   # inference servers
pip install -U huggingface_hub                 # the `hf` command
```

## 2. Check the installation (no GPU, about 1 minute)

```sh
make test     # unit tests, no data, no models
make smoke    # the whole pipeline on 5 questions with BM25 and a fake LLM
```

Both should pass. `make smoke` writes a run folder under `data/runs/` with `predictions.jsonl` and `metrics.json`.

## 3. Get the data

**Course data.** Put the two files from the course Drive folder ("RAG Assignment") in `data/raw/`: `Dataset.zip` (16,441 law texts) and `res.csv` (2,745 question/answer rows). Then:

```sh
make data     # unzip, clean, and write the fixed seeded split: test 200 / dev 300 / train 2,245
```

The split is deterministic. `dataset/split.json` in the submission holds the question ids of each split, so you can check yours is identical.

**Extra corpus and training data.** The final system also retrieves from central-government laws since 2000 and from the laws the answers cite. Rebuilding them (`make vbpl-cited`, `make web`) scrapes public portals and takes hours, and the portals change. Download the exact snapshot used in the report instead:

```sh
scripts/ops/fetch_artifacts.sh              # data/processed (corpus), data/train (SFT + DPO data), data/models/dpo_a2 (weights)
scripts/ops/fetch_artifacts.sh --no-model   # data only
```

| Folder | Content |
|---|---|
| `data/processed/` | `qa.parquet` (questions and references), `split.json`, `gold.parquet` (cited articles), and the law texts, one file per document: `corpus/` (course), `vbpl_cited/` (laws since 2000 and every cited law), `web/` (cited laws from public portals) |
| `data/train/q35_v3b/` | SFT rows (prompt as seen at inference + reference answer), decontaminated against test |
| `data/train/pref_v3b/`, `data/train/pref_dpoa/` | DPO pairs of rounds 1 and 2 |

## 4. Run the final system and reproduce the main result

Start three vLLM servers (one GPU is enough: they share it):

```sh
V=~/vllm_env/bin/vllm
$V serve data/models/dpo_a2 --port 8000 --gpu-memory-utilization 0.45 --dtype bfloat16 --max-model-len 32768 --enable-prefix-caching --seed 0 &
$V serve AITeamVN/Vietnamese_Embedding --port 8100 --runner pooling --gpu-memory-utilization 0.15 --dtype bfloat16 &
$V serve AITeamVN/Vietnamese_Reranker  --port 8200 --runner pooling --gpu-memory-utilization 0.15 --dtype bfloat16 &
```

Then run the evaluation (the first run builds the FAISS index of one passage per Điều, 12 GB under `data/index/`; this takes a while, later runs reuse it):

```sh
M=$PWD/data/models/dpo_a2
make run EXP=configs/exp/A5_v3b.yaml   ARGS="--set backend=gpu --set generate.model=$M --set generate.shots=2 --set split=test"   # greedy
make run EXP=configs/exp/A6_mbrv3b.yaml ARGS="--set backend=gpu --set generate.model=$M --set generate.shots=2 --set split=test"  # MBR-8
```

Each run writes `data/runs/<exp>__<hash>/` with `config.yaml`, `predictions.jsonl` and `metrics.json` (cosine, Jaccard, token overlap, BLEU-1, ROUGE-L, the course notebook's metrics), and a row in `data/runs/results.tsv`. Drop `--set split=test` for the dev set (300 questions).

Expected: ROUGE-L about .643 (greedy) and .652 (MBR-8). vLLM greedy decoding is not bit-identical between runs, so expect ±.003.

To compare two runs with a paired bootstrap: `make compare A=data/runs/<a> B=data/runs/<b> METRIC=rouge_l`.

## 5. The question-answering service

A FastAPI service with the same pipeline, packaged with Docker Compose (four containers: API, LLM, embedder, reranker):

```sh
PYTHONPATH=src uv run python -m cli export-shots        # few-shot pool for the container
docker compose -f docker/compose.yml build api
GPU=0 docker compose -p lawrag-qa -f docker/compose.yml up -d
curl -s localhost:8080/health
curl -s -X POST localhost:8080/ask -H 'Content-Type: application/json' \
  -d '{"question":"Mức phạt đối với hành vi không đội mũ bảo hiểm khi đi xe máy là bao nhiêu?"}'
```

The answer comes with its five sources and the time spent in each stage. When no passage passes the reranker threshold the service refuses instead of guessing. Streaming, an OpenAI-compatible endpoint, metrics and the two-GPU layouts are described in `docs/ops/docker.md`.

## 6. Retrain the model from scratch

On one 46 GB GPU: SFT about 1.3 h, each DPO round about 1 h including sampling.

```sh
# SFT data (decontaminated against test) and LoRA SFT
PYTHONPATH=src uv run python -m cli sft-data configs/exp/A3_Q3.yaml --name q35_v3b --k-range 3,10 --oracle 0.7 --shots 0,1,2,3 --refuse 0.2
python src/train/sft/sft_lora.py --data data/train/q35_v3b --base Qwen/Qwen3.5-2B --out data/models/q35_v3b --epochs 3 --max-length 12288 --batch 1 --accum 16

# DPO round 1: serve the SFT model on port 8000, sample 8 answers per training prompt, pair best vs worst by ROUGE-L, train
python scripts/train/pref_sample.py --data data/train/q35_v3b --out data/train/pref_v3b/cands.jsonl --model data/models/q35_v3b --base-url http://localhost:8000/v1
python scripts/train/build_pairs.py data/train/pref_v3b/cands.jsonl data/train/pref_v3b/pairs.jsonl --min-margin 0.1
python src/train/preference/dpo.py --pairs data/train/pref_v3b/pairs.jsonl --base data/models/q35_v3b --out data/models/dpo_a --lr 2e-5 --beta 0.1

# DPO round 2: the same three steps from data/models/dpo_a -> data/train/pref_dpoa -> data/models/dpo_a2
```

Then evaluate `data/models/dpo_a2` as in section 4. Seeds change the result by up to about .016 ROUGE-L (report, "Seed dependence").

## 7. Other experiments in the report

Every experiment is one YAML file in `configs/exp/` (only its differences from `configs/base.yaml`), and every GPU job sequence is a queue file in `configs/queues/` (indexed by topic in `configs/queues/*/INDEX.md`). `docs/CHANGELOG.md` lists every step in order with its numbers, and the report cites the step for each table.

| Report section | Configs / scripts |
|---|---|
| Course baseline | `configs/exp/B0_baseline.yaml` |
| Retrieval ablations, chunking | `configs/exp/A2*.yaml`, `configs/queues/retrieval/` |
| SFT variants, learning curve | `configs/queues/sft/` |
| DPO, ablations, replications | `configs/queues/dpo/` |
| Final pre-declared configuration | `configs/queues/final/`, `docs/research/final-run-plan.md` |
| Scaling to 4B / 9B | `configs/queues/final/q4b_g1.txt`, `configs/queues/final/q9b_g0.txt` |
| Load tests | `scripts/serve/serve_bench.sh`, `scripts/serve/load_test.py` |

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| vLLM fails with a FlashInfer error | `export VLLM_USE_FLASHINFER_SAMPLER=0` |
| Out of GPU memory with three servers on one card | lower `--gpu-memory-utilization` of the LLM, or put embedder and reranker on a second GPU |
| `make run` cannot reach the LLM | check `curl localhost:8000/v1/models`; `generate.model` must be the same path the server was started with |
| Index build killed (out of RAM) | it needs about 30 GB RAM at peak; close other jobs |
| Scores differ slightly from the report | vLLM greedy is not deterministic (±.003); retrained models vary with the seed (up to .016) |
