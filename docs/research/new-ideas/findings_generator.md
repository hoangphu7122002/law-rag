# Generator-side ideas (Qwen3.5-2B, ROUGE-L, templated legal answers)

Status: 4 web searches, search-snippet level only. ALL numbers UNVERIFIED (no full paper read). Excluded: MBR decode, SFT+few-shot/refusal, GRPO, on-policy distill 9B, ReAct SFT, merging.

## 1. MBR/BoN distillation via preference pairs (DPO on ROUGE-ranked candidates)
- What: sample N (8-16) answers/query, score each by ROUGE-L vs REFERENCE (we have gold, stronger than MBR pseudo-ref) or vs other samples. chosen=best, rejected=worst. DPO/SimPO -> greedy matches MBR/BoN quality.
- Evidence: "Better Instruction-Following Through MBR" (ICLR25): DPO on MBR-chosen pairs makes greedy ≈ base+MBR; holds up to 70B. https://arxiv.org/html/2410.02902v3 . Caveat: ROUGE-MBR gave only +1.33 AlpacaEval win-rate vs +3.62 for Prometheus judge (chat task; ours is ROUGE metric itself so utility matches metric). MBR>BoN generally. NMT precursor Yang et al 2023 https://arxiv.org/pdf/2311.08380v1 (UNVERIFIED numbers). Iterative MBR distill 2026: https://arxiv.org/pdf/2512.07540
- Cost: low. N=8 sampling 2B vLLM on train set + 1-2 DPO epochs LoRA/full on 2B. ~1 GPU-night.
- Risks: length blow-up (DPO length bias); chosen-logp decline; gold-ref ROUGE chosen may be near-duplicate of SFT target (little margin) -> filter pairs by ROUGE gap >= 0.05. Overlaps GRPO conceptually (offline vs online) but not excluded.
- TRL: DPOTrainer, dataset {prompt, chosen, rejected}; init from SFT ckpt; beta 0.1, lr 5e-7 (full) / 5e-6 (LoRA). Add rpo_alpha (SFT term on chosen) to avoid drift.
- Cheapest test: 500 queries x 8 samples from current SFT, pairs gap>=0.05, DPO 1 epoch, eval greedy ROUGE-L dev vs SFT and vs MBR(7).

## 2. SimPO / length-controlled preference loss
- What: reference-free, avg-logprob reward + margin (length-normalized). SimPO +6.4 AlpacaEval2 over DPO (7-9B). https://arxiv.org/html/2405.14734v3
- 1-4B caution: controlled study 2603.19335 (Qwen2.5, GSM8K): SimPO worst at small scale, best at 7B. https://arxiv.org/pdf/2603.19335.pdf . No ROUGE/summarization 1-4B head-to-head found (gap). LMPO length-controlled variant: https://web3.arxiv.org/pdf/2502.14643
- Use as A/B against DPO inside idea 1; also ORPO (single stage, no ref, in TRL ORPOTrainer) and KTO (unpaired, KTOTrainer, needs only good/bad labels e.g. ROUGE>thr) as cheaper variants. No evidence on 1-4B ROUGE.
- Cheapest test: same pair set as #1, run DPO vs SimPO(loss_type="simpo", cpo_alpha) , compare ROUGE-L + mean length.

## 3. Citation-aware training (right Điều among retrieved)
- SelfCite (Meta): reward = context-ablation probability drop; BoN + SimPO; citation F1 up to +5.3 on LongBench-Cite. https://arxiv.org/html/2502.09604v3 (model sizes unverified, 8B-class).
- Trust-Align (ICLR25): 19K pos/neg pairs, DPO; Llama-3-8B TRUST-SCORE +12.6 ASQA/+36 QAMPARI/+17.7 ELI5 vs FRONT. https://arxiv.org/pdf/2409.11242.pdf (UNVERIFIED; includes refusal pairs).
- Ours: citation is deterministic -> cheap synthetic negatives: rejected = same answer with wrong khoản/Điều/Thông tư taken from a distractor retrieved passage (or quote of wrong article). chosen = gold. Teaches article selection directly; ROUGE gain comes from correct quote.
- Cost: no sampling needed, templated corruption. Risk: model learns surface cues to corruption only; mix with on-policy negatives.
- TRL: DPOTrainer. Cheapest test: 2k pairs w/ distractor-article negatives, DPO 1 epoch, measure citation exact-match + ROUGE-L.

## 4. Template/prefix-forced decoding
- Force prefix "Căn cứ tại khoản N Điều M ..." where N/M/Thông tư chosen by retriever top-1 (or reranker) metadata, model continues only the "quy định ..." body. Search found no paper (gap); standard in HF via prefill/ assistant prefix, vLLM guided decoding (xgrammar/regex) for the opening only.
- Gain mechanism: removes citation errors and opening variance; ROUGE-L opening tokens ~8-12 of answer. Risk: wrong top-1 retrieval locks error; cap by using retriever score threshold, else free-gen.
- Cost: ~zero training; inference-time. Can also train SFT with prefix loss-masked (TRL assistant_only/completion_only_loss).
- Cheapest test: dev eval, prefix from gold article ids (oracle) vs retriever top-1 vs free. Oracle gap = upper bound of citation ideas.

## 5. Length/format control
- Tune max length & length penalty; DPO/SimPO length normalization (idea 2); ROUGE-L F1 peaks when length ~ reference. Cheapest: grid length_penalty / max_new_tokens / EOS bias on dev with current ckpt; check mean len ratio vs reference. No external evidence needed. Add length-gap filter on DPO pairs (|len diff| small) to avoid length-only learning.

## 6. Test-time scaling: BoN + reranker (beyond MBR)
- Small models gain from BoN: Qwen2.5-0.5B BoN w/ Nemotron-70B RM win rate 0.62->0.935 (class project, UNVERIFIED) https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Report__1_.pdf ; MT: 3B/7B N=1024 reranking beats larger N=1 (UNVERIFIED) https://www.themoonlight.io/review/investigating-test-time-scaling-with-reranking-for-machine-translation ; 1B vs 405B math w/ PRM https://www.emergentmind.com/papers/2502.06703 ; SCATR cheap learned scorer https://arxiv.org/pdf/2604.16535
- Ours: train a small ROUGE-predictor/reranker (cross-encoder on (query, passages, candidate) -> ROUGE-L vs gold) to pick among N; or score candidate by quote overlap with retrieved article (no learning): ROUGE-L(candidate, top-1 article) as proxy reward. Beyond MBR since uses passage, not consensus.
- Cost: N samples at inference (latency x N); check competition time limits.
- Cheapest test: on dev, N=8 samples, oracle-best ROUGE (upper bound) vs MBR vs candidate-vs-article-overlap selector.

## 7. Self-consistency distillation
- Subsumed by #1 (MBR = consensus). Alternative: SFT on MBR-picked outputs (RFT/"best-of-N SFT"), simplest, no DPO. Test as ablation of #1. UNVERIFIED beyond MBR paper.

## Gaps
No 1-4B ROUGE-specific evidence for DPO/SimPO/KTO/ORPO; no paper on prefix-forced legal citation openings; SelfCite/Trust-Align model sizes unverified. Do not cite numbers before reading full papers.
