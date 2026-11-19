# Study path: learn while building law-rag (small LLM RAG + post-training)

All URLs HTTP-checked (200) on 2026-11-20. Times = focused hours. Total ~110-140 h; do 1 item per ~1 week, in order. Always do the exercise right after the reading. Pin library versions: TRL moves fast (GKD/GOLD are under `experimental`; import paths change by version).

Rule: read "first" bit, build exercise, then read rest only as needed.

## 1. Transformers + LoRA/PEFT basics (~10 h)
- HF LLM Course: https://huggingface.co/learn/llm-course (ch.1-3 + ch.11 LoRA)
- PEFT quicktour: https://huggingface.co/docs/peft/quicktour ; smol-course (LoRA unit): https://github.com/huggingface/smol-course
- LoRA paper (skim): https://arxiv.org/abs/2106.09685
- Read first: LLM course ch.1-2 (tokenizer, chat template, generate), then PEFT quicktour.
- Exercise: load Qwen3.5-2B, apply chat template to 5 law-rag prompts, inspect tokens/loss mask; wrap with `LoraConfig` (r=16, all linear), print trainable params, save/load/merge adapter, compare outputs base vs merged (ties to src/train weight merging).

## 2. SFT with TRL (~8 h)
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer (read: dataset formats, `assistant_only_loss` / completion-only loss, packing, PEFT)
- smol-course SFT unit (same repo as above).
- Read first: dataset format + completion-only loss section.
- Exercise: SFT on law-rag (question+context -> answer); run twice, full-sequence loss vs completion-only/assistant-only; compare ROUGE-L on dev + loss curves. Verify the mask by decoding labels != -100.

## 3. Preference optimisation: DPO, SimPO, KTO (~10 h)
- DPO paper: https://arxiv.org/abs/2305.18290 (read sec. 3-4 only first)
- TRL DPOTrainer: https://huggingface.co/docs/trl/dpo_trainer ; KTO: https://huggingface.co/docs/trl/kto_trainer
- SimPO: https://arxiv.org/abs/2405.14734 (reference-free, length-normalised; TRL exposes it via DPO `loss_type`, check your version)
- Read first: DPO derivation intuition (implicit reward = beta*log ratio), then TRL `loss_type` list.
- Exercise: sample N=8 candidates per question from SFT model, rank by ROUGE-L vs gold (MBR / best-vs-worst), build chosen/rejected pairs, run DPO then SimPO; compare vs SFT baseline with paired bootstrap (item 8). Watch length bias: log mean output length.

## 4. RL for LLMs: GRPO (~14 h)
- DeepSeekMath (introduces GRPO): https://arxiv.org/abs/2402.03300 (sec. 4)
- DeepSeek-R1: https://arxiv.org/abs/2501.12948 (R1-Zero, rule-based rewards)
- TRL GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer (reward funcs, vLLM colocate/server mode, loss types)
- Practical: HF Open-R1 blog https://huggingface.co/blog/open-r1
- Read first: TRL GRPO doc, then DeepSeekMath GRPO section (group-normalised advantage, no critic).
- Exercise: GRPO with reward = ROUGE-L vs gold (+ small format/length penalty), vLLM colocate on 1 GPU, G=8, start from SFT checkpoint; log reward mean/std, KL, completion length. Fallback per memory: only if SFT plateaus. Test reward fn first (TDD).

## 5. Distillation incl. on-policy (~10 h)
- Thinking Machines "On-Policy Distillation": https://www.thinkingmachines.ai/blog/on-policy-distillation/
- GKD paper (Agarwal et al.): https://arxiv.org/abs/2306.13649 ; TRL GKDTrainer: https://huggingface.co/docs/trl/gkd_trainer ; GOLD (cross-tokenizer): https://huggingface.co/docs/trl/gold_trainer
- Read first: TM blog (SFT vs RL vs on-policy distill table), then GKD sec. 3 (lambda = on-policy fraction, JSD beta).
- Exercise: (a) off-policy: SFT on Haiku-generated answers (approved teacher, decontaminate); (b) on-policy: GKDTrainer with a larger Qwen teacher (same tokenizer) on the box, lmbda=1.0; compare (a) vs (b) on dev ROUGE-L. Note: Haiku has no logits, so token-level KL needs an open teacher.

## 6. RAG, retrieval, reranking (~16 h)
- LlamaIndex starter + local: https://developers.llamaindex.ai/python/framework/getting_started/starter_example_local/ ; understanding guide: https://developers.llamaindex.ai/python/framework/understanding/
- BEIR: https://arxiv.org/abs/2104.08663 ; MTEB: https://arxiv.org/abs/2210.07316 ; RAG original: https://arxiv.org/abs/2005.11401
- sentence-transformers training: https://www.sbert.net/docs/sentence_transformer/training_overview.html
- Read first: LlamaIndex starter (local), then BEIR sec. 1-4 for metrics (nDCG@10, Recall@k).
- Exercise: build index over law corpus with local embeddings; measure Recall@k / nDCG@10 on dev with gold passages; add cross-encoder reranker; fine-tune embedder with MultipleNegativesRankingLoss + hard negatives; report delta. Retrieval quality caps generation, measure it separately.

## 7. Serving and efficiency (~10 h)
- vLLM docs: https://docs.vllm.ai/en/latest/ ; speculative decoding: https://docs.vllm.ai/en/latest/features/spec_decode.html
- Spec decoding paper: https://arxiv.org/abs/2211.17192 ; GPTQ: https://arxiv.org/abs/2210.17323 ; AWQ: https://arxiv.org/abs/2306.00978
- Read first: vLLM quickstart + offline inference, then paged attention / continuous batching concepts.
- Exercise: serve base vs LoRA (vLLM `--enable-lora`) on one L40S; benchmark tokens/s and p95 latency at concurrency 1/8/32; try AWQ/FP8 quant and check ROUGE-L drop; use vLLM for MBR candidate sampling throughput.

## 8. Evaluation (~12 h)
- ROUGE original: https://aclanthology.org/W04-1013/ (pitfalls: lexical overlap, length/recall bias, tokenisation for Vietnamese, ROUGE-L variant choice summary vs sentence)
- RAGAS paper: https://arxiv.org/abs/2309.15217 ; docs: https://docs.ragas.io/en/stable/
- Paired bootstrap: Berg-Kirkpatrick et al. 2012 https://aclanthology.org/D12-1091/
- Read first: bootstrap paper sec. 2-3, then RAGAS metric list (faithfulness, context precision/recall).
- Exercise: write `paired_bootstrap(scores_a, scores_b, n=10000)` returning delta, 95% CI, p; tests first. Re-evaluate every earlier item's gain with it. Add faithfulness check (judge) alongside ROUGE; show cases where ROUGE up but answer wrong/unfaithful.

## 9. Agentic / ReAct / tool-use training (~14 h)
- ReAct: https://arxiv.org/abs/2210.03629 ; Search-R1: https://arxiv.org/abs/2503.09516 (code: https://github.com/PeterGriffinJin/Search-R1)
- HF tool-calling / chat templates: https://huggingface.co/docs/transformers/chat_extras
- Read first: ReAct sec. 2-3 (Thought/Action/Observation), then Search-R1 retrieved-token masking + outcome reward.
- Exercise: ReAct loop with the retriever as `search` tool; build trajectories, SFT with assistant-only loss (existing ReAct S1 pipeline), mask observation tokens; then optional GRPO with outcome (ROUGE) reward à la Search-R1. Compare vs single-shot RAG with bootstrap.

## Caveats
- Search surfaced TM blog figures only second-hand; read the blog itself.
- TRL API (GKD/GOLD experimental, SimPO as DPO loss_type) is version-dependent: check installed version docs.
- URLs checked for HTTP 200 only; content not fully read.
