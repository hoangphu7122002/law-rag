# New ideas + study path (2026-11-20) — synthesis of findings_generator.md, findings_retrieval_efficiency.md, findings_study_path.md

All numbers come from search snippets (UNVERIFIED). URL status checked (HTTP 200) for the study path, pages not read.

## Ranked new ideas for this project (cheapest evidence first)
| # | Idea | Why it fits (our facts) | Cheapest test | Training? |
|---|---|---|---|---|
| 1 | **Prefix-forced opening "Căn cứ tại khoản N Điều M …"** | references start with it; wrong cited Điều costs ~.30 ROUGE-L | dev: prefix from gold ids (upper bound) vs retriever top-1 vs free generation | no |
| 2 | **Cascade: greedy first, MBR only when uncertain** | MBR8 = +.03 but 8x cost; latency is an optimisation axis | bucket existing MBR vs greedy dev outputs by reranker top-1 margin | no |
| 3 | **Adaptive k by reranker margin** | prefill dominates; fewer distractors | sweep margin threshold on dev: ROUGE-L vs prompt tokens | no |
| 4 | **Oracle of passage selection** (generate once per top-5 Điều, pick best) | decides whether reranker-in-the-loop / RL-over-selection is worth anything | ~2k short generations; gap < .02 → drop the family | no |
| 5 | **DPO on ROUGE-ranked candidates** (MBR/best-of-N distillation into greedy) | MBR already gives 7 samples per question | 500 q x 8 samples, pairs with gap >= .05, TRL DPOTrainer 1 epoch (+ rpo_alpha) | yes (1 h) |
| 6 | **Citation negatives for DPO** (same answer citing a wrong Điều from a distractor) | targets our biggest error | 2k synthetic pairs, DPO 1 epoch, measure Điều match + ROUGE-L | yes |
| 7 | **Lossless speedups**: vLLM FP8, prompt-lookup speculation, prefix caching | latency axis; answers copy Điều text | bf16 vs FP8 dev ROUGE-L + p50/p95 | no |
| 8 | Haiku listwise rerank check -> distillation | tests teacher quality | 300 dev q where gold in top-10 but not top-1 | optional |
| 9 | SimPO / ORPO / KTO instead of DPO | small-scale evidence weak | same pair set as #5 | yes |
Not worth now: RL over passage selection, synthetic QA per Điều, hard-negative variants, curriculum, paraphrase augmentation.

## Study path (read -> do in this repo) — ~100 h in total, do in order, one exercise per step
| # | Topic | Read first | Time | Do in law-rag |
|---|---|---|---|---|
| 1 | Transformers + LoRA/PEFT | HF LLM course; PEFT quicktour; smol-course | 10 h | load Qwen3.5-2B, inspect chat template/tokens, LoRA, save/merge/compare |
| 2 | SFT with TRL | TRL sft_trainer docs | 8 h | completion-only vs full-sequence loss, check label mask (src/train/sft_lora.py, react_sft.py) |
| 3 | Preference tuning | DPO paper 2305.18290; TRL dpo/kto docs; SimPO 2405.14734 | 10 h | idea #5 on MBR candidates |
| 4 | GRPO | DeepSeekMath 2402.03300; R1 2501.12948; TRL grpo_trainer; Open-R1 blog | 14 h | read src/train/grpo.py reward `score`, unit-test it, rerun a small GRPO |
| 5 | Distillation / on-policy | Thinking Machines "On-Policy Distillation" blog; GKD 2306.13649; TRL gkd/gold docs | 10 h | docs/research/on-policy-distill/findings.md; Haiku gives no logits -> open teacher (Qwen3.5-9B) |
| 6 | RAG + retrieval + rerank | LlamaIndex local starter; BEIR 2104.08663; MTEB 2210.07316; sbert.net training | 16 h | measure Recall@k/nDCG separately from generation (src/evaluate), embed-bench |
| 7 | Serving + efficiency | docs.vllm.ai; speculative decoding 2211.17192; GPTQ 2210.17323; AWQ 2306.00978 | 10 h | idea #7, report p50/p95 per stage |
| 8 | Evaluation | ROUGE paper W04-1013; RAGAS 2309.15217; paired bootstrap D12-1091 | 12 h | embed_bench.paired_bootstrap, data/cmp3.py: re-test earlier gains |
| 9 | ReAct / tool use | ReAct 2210.03629; Search-R1 2503.09516; HF chat_extras | 14 h | src/generate/react.py, src/train/react_*.py, docs/notes/16-react-training.md |
Caveats: TRL GKD/GOLD trainers are `experimental` (import paths change; check the installed version); SimPO is a DPO `loss_type` in TRL (unconfirmed); Thinking Machines numbers were second-hand — read the blog.
