# On-policy distillation for small tool-using agents: findings

Date 2026-11-19. Source: ~9 web searches/fetches. `[UNVERIFIED]` = from search snippets/secondary summaries only.

## 1. Thinking Machines "On-Policy Distillation"
URL: https://thinkingmachines.ai/blog/on-policy-distillation/
- Student samples rollouts; teacher scores each token. Per-token advantage = -(log p_student - log p_teacher) (negative reverse KL), trained with RL importance-sampling loss. One-line change on RL code with KL reg.
- Discount factor 0 (>0 did not help). Sampled-token estimate only; NO top-k / full-vocab logits in any experiment.
- Teacher needs just one forward pass (`compute_logprobs`) over student's sampled tokens.
- Recipe numbers: ~150 steps (~77K prompts), 64 prompts x 4 samples = 256 rollouts/batch. LoRA rank 32/128 and full FT both used. Student Qwen3-8B-Base (after 400K-prompt SFT), teacher Qwen3-8B/32B.
- LR, alpha, KL coef NOT given in post (fetched summary). Multi-turn / loss masking NOT discussed.
- Compute: ~9x FLOPs vs SFT to 2M prompts (~18x GPU-hours); ~30x counting teacher sampling for new task; self-distill 50-100x.
- Results: AIME'24 SFT-400K 60% -> 70% in ~150 steps (teacher 74-ish). Qwen3 report: off-policy 55.0 -> +RL 67.6 (17,920 GPU-h) vs +OPD 74.4 (1,800 GPU-h). LoRA r32: gap to full FT 13% (SFT) -> 6% (distill). Self-distill recovered in <10 steps vs RL ~70.
- Note: post says cookbook now uses Qwen3.5-9B-Base student / Qwen3.5-9B teacher.

## 2. TRL support (v1.18.x docs)
URLs: https://huggingface.co/docs/trl/en/distillation_trainer.md , https://huggingface.co/docs/trl/async_distillation_trainer , https://huggingface.co/docs/trl/gkd_trainer
- Old `GKDTrainer` (experimental): `lmbda` = fraction on-policy student data (0 = supervised, 1 = fully on-policy), `beta` = JSD interpolation (0 fwd KL, 1 rev KL; config default 0.5 [UNVERIFIED]), `seq_kd` = SFT on teacher generations. Teacher loaded locally.
- NEW `trl.DistillationTrainer` (top-level): always on-policy (no lmbda), generalized JSD via `beta` (default 1.0 = reverse KL), full-vocab chunked loss (teacher dense dist never materialized), teacher loaded LOCALLY (`teacher_model`), must share student vocab. Student gen via vLLM colocate/server. PEFT ok (no adapters on lm_head; use modules_to_save). Defaults: lr 1e-6, max_completion_length 512, temperature 1.0, gradient_checkpointing on. LoRA CLI example lr 2e-4, r64, alpha16.
- **Agent training supported**: `tools=[python funcs]`, `max_tool_calling_iterations`; tool-result tokens masked from loss; needs prefix-preserving chat template (TRL auto-patches Qwen3 family; check Qwen3.5 [UNVERIFIED]). Sync tools only. Logs tools/call_frequency, failure_frequency.
- `AsyncDistillationTrainer` (experimental): teacher ONLY via external vLLM server (`teacher_server_urls`), student on own vLLM server w/ NCCL weight sync, trainer on 3rd GPU set; FSDP2 only, needs transformers>=5.2, kernels. Teacher scored via `/v1/completions` + `prompt_logprobs` over raw token ids; only top-k (`teacher_top_k` default 8; suggest 16-64) + realized token + tail bucket transmitted. Teacher must share tokenizer. beta default 0.0. LoRA sync via `--enable-lora` + VLLM_ALLOW_RUNTIME_LORA_UPDATING=1. Tool/multi-turn support NOT mentioned in async docs (grep found none) -> assume unsupported.
- Teacher server flags: `vllm serve <teacher> --logprobs-mode processed_logprobs --max-logprobs -1`.
- Reality for us (2x L40S): sync DistillationTrainer = student(+LoRA)+vLLM colocate+9B bf16 teacher (~18GB) locally; fits tight on 46GB if student on GPU0 w/ vLLM util ~0.3 and teacher on GPU1? Single-process teacher placement is NOT per-GPU controllable by docs [UNVERIFIED]. Alternative: custom loop (below).

## 3. MiniLLM / DistiLLM / GKD
- GKD (Agarwal et al.) https://arxiv.org/abs/2306.13649 : train on student's own samples with teacher feedback, flexible divergence (JSD/fwd/rev KL); lmbda mix of on/off-policy.
- MiniLLM https://arxiv.org/abs/2306.08543 : reverse KL via policy gradient + single-step decomposition, teacher-mixed sampling, length normalization (stabilizers; details [UNVERIFIED]).
- DistiLLM https://arxiv.org/abs/2402.03898 : skew KL/skew RKL (alpha~0.1 [UNVERIFIED]) + adaptive off-policy w/ replay buffer; claims 2.5-4.3x speedup vs MiniLLM/GKD [UNVERIFIED, secondary]. DistiLLM-2 https://arxiv.org/abs/2503.07067 (contrastive).
- Takeaway: sampled-token reverse KL (TM) == MiniLLM-style, simplest; skew/JSD only needed if full dist used.

## 4. Agentic distillation + failure modes
- Agent Distillation (Kang et al.) https://arxiv.org/abs/2505.17612 : SFT on teacher reason-act-observe trajectories; "first-thought prefix" (teacher CoT first step used as trajectory prefix) improves teacher traj quality; self-consistent action gen at test time. 0.5-3B students match next-tier-larger CoT-distilled 1.5-7B [from abstract/snippets]. Code github.com/Nardien/agent-distillation. Off-policy only.
- SOD (step-wise OPD for small agents) https://arxiv.org/abs/2605.07725 : OPD on tool-integrated reasoning -> erroneous tool calls cascade, student-teacher gap widens, teacher unreliable on out-of-distribution prefixes; fix = per-step reweighting by divergence. [UNVERIFIED beyond snippet; arXiv ids are post-cutoff, verify]
- UOPD https://arxiv.org/abs/2609.34036 : teacher intervention at high-uncertainty turns in multi-turn agents. [UNVERIFIED]
- Top-K truncation hazard https://www.alphaxiv.org/abs/2607.07050 : teacher top-32 covers 99.99% mass but lacks the tool-call token on 0.4% of prompts -> tool-call drift. Implication: ALWAYS include realized token (sampled-token estimator does) and avoid pure top-k forward KL on call/no-call decision tokens. [UNVERIFIED]
- Qwen3 strong-to-weak https://arxiv.org/abs/2505.09388 (sec 4.5): off-policy distill first, then on-policy logit matching. SFT/warm start before OPD is standard. TM also starts from SFT.
- "Prefix Teach, Suffix Fade" https://arxiv.org/abs/2605.13643 : teacher signal degrades along long student rollouts (local teachability collapse). [UNVERIFIED]
- Search-R1 / ToolRL: masking retrieved/tool-output tokens from loss is standard (TRL DistillationTrainer does this). Not fetched this round; [UNVERIFIED].
- Failure modes checklist:
  - Tokenizer mismatch: none for Qwen3.5-9B -> 2B (same tokenizer; verify identical vocab incl. special tokens/padding of embed matrix; sizes can differ even if tokenizer same).
  - Tool outputs: mask from loss, but teacher must still SEE them in context (score full sequence, loss only on student tokens).
  - Teacher hallucination: teacher logprob on student's wrong-but-fluent tokens can still be high; teacher not grounded. Add outcome reward/filter (answer correct or cites right article) or mix with GRPO.
  - Chat-template / thinking mode: teacher must be scored with the SAME rendered text (same system prompt, thinking flag). Template mismatch = garbage logprobs.
  - Collapse/length: reverse KL is mode-seeking and can shorten/loop outputs; track completion length, clipped_ratio, entropy, tool-call freq. Cap max_completion_length and max tool iterations.
  - Rollout/trainer logprob mismatch (vLLM vs HF, bf16): compute student logprobs in trainer, or use TIS/clip.

## 5. Practical recipe (2B LoRA, 2x L40S)
- Warm start: SFT'd student (we have SFT checkpoints). Do not OPD from raw base.
- Batch: 64 prompts x 4 samples (TM) -> for us 32 prompts x 4 = 128 rollouts/step is realistic (multi-turn rollouts long). 
- Steps: 100-200 (TM ~150). LR: TRL defaults 1e-6 full-FT; LoRA 1e-5 to 2e-4 range (TRL LoRA CLI 2e-4; common RL-LoRA ~1e-5..5e-5). Suggest start 5e-5, r=32-64 [judgment, not sourced].
- KL coef: none in TM/TRL (teacher IS the KL). Discount 0. No separate reference KL.
- Temperature 1.0 sampling (matches loss), top_p 1.
- Loss: advantage_t = log p_T(x_t) - log p_S(x_t) (detached), PG loss -A*ratio on student-generated tokens only; mask tool-output tokens and prompt. Optionally clamp advantage to [-5,5] [judgment].
- Teacher logprobs from vLLM:
  - OpenAI `/v1/completions` with `prompt` = list of token ids (prompt+response), `max_tokens=1` (or 0), `prompt_logprobs=0` (or 1): returns logprob of each realized prompt token (vLLM always includes the realized token). Works over HTTP. `echo` semantic is vLLM's prompt_logprobs [UNVERIFIED].
  - Arbitrary token ids at each position: offline `LLM.generate(TokensPrompt, SamplingParams(prompt_logprob_token_ids=..., prompt_logprob_start=len(prompt)-1, max_tokens=1))` -> `prompt_token_id_logprobs`; HTTP `/inference/v1/generate` returns base64 npy. Not on OpenAI endpoints. https://docs.vllm.ai/en/stable/training/prompt_token_id_logprobs/
  - Serve with `--logprobs-mode processed_logprobs` (or raw) consistent w/ temperature; `--max-logprobs` as needed; disable kv-sharing-fast-prefill.
  - For sampled-token only (TM style) the plain `prompt_logprobs=0` is enough and cheapest.
- Implementation options:
  1. TRL `DistillationTrainer` + tools (native multi-turn, loss mask, full-vocab JSD; teacher local). Easiest if GPU placement works; uses full-vocab reverse KL (beta=1) rather than sampled token.
  2. Custom loop (TM style): vLLM student rollouts via our ReAct env -> token ids + assistant mask -> teacher vLLM server `prompt_logprobs` -> HF LoRA update. More work, supports external teacher + our tool env exactly.
  3. AsyncDistillationTrainer: external teacher but no documented tool support.
- GPU layout idea: GPU0 teacher vLLM (9B bf16 ~18GB + KV), GPU1 student vLLM (2B) + LoRA trainer in turn (sleep mode) or colocated.
- Eval gates: dev-set answer accuracy/citation F1 every 25 steps, tool-call validity rate, mean length. Stop if length collapses or tool-call freq drifts.
