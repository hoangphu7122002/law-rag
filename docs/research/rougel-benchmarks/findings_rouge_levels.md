# ROUGE-L levels in legal QA / RAG (web research, 5 calls)

Verified = read page text via fetch. UNVERIFIED = search snippet only.

## (a) Reported numbers
| Source | Task | ROUGE-L | Status |
|---|---|---|---|
| VLQA (Vietnamese statute QA) https://arxiv.org/html/2507.19995v1 | extractive (PhoBERT, mBERT, XLM-R, LEGAL-BERT) | 0.227-0.268 | verified |
| same | fine-tuned generative ViT5 / BARTpho | 0.321 / 0.332 | verified |
| same | LLM 0/2-shot: Llama3.1-8B 0.38; Qwen2.5-7B 0.42; Llama3-8B 0.50; Qwen2.5-14B 0.52; DeepSeek-V3 0.52-0.56; GPT-4o 0.59-0.60; GPT-4o-mini 0.60-0.62 | 0.38-0.62 | verified |
| ViBidLAQA_base (ViT5-base, VN bidding law, Claude-generated train data, expert test) https://huggingface.co/ntphuc149/ViBidLAQA_base | abstractive QA | 65.72 (x100) | UNVERIFIED (model card, snippet) |
| Hu et al. COLING 2025 (Chinese statute QA) https://arxiv.org/pdf/2501.06521 | zero-shot LLMs: Llama3.1-70B 0.221, 405B 0.249, GPT-4o-mini 0.254, GPT-4o 0.266 ("ROUGE", variant unclear) | 0.22-0.27 | UNVERIFIED |
| same | SFT+HIPO GLM4-9B: ROUGE-L +126.7% rel. over base (absolute not seen) | n/a | UNVERIFIED |
| CLERC https://arxiv.org/pdf/2406.17186v1 | long-form legal analysis; GPT-4o highest ROUGE-F but most hallucination | low (abs. not seen) | UNVERIFIED |
| LegalBench-RAG ext. https://arxiv.org/pdf/2508.13107v2 | generation scored by ROUGE-Recall, BERTScore, RAGAS | abs. not seen | UNVERIFIED |
| Hybrid RAG judicial QA https://arxiv.org/pdf/2511.01668 | ROUGE-L with beta=1.2 (recall-biased) | abs. not seen | UNVERIFIED |
| ALQAC 2024/25 https://sites.google.com/view/ALQAC-2025 | scored by accuracy / F2, not ROUGE-L | n/a | UNVERIFIED |
| LegalBench-RAG orig. https://arxiv.org/abs/2408.10343 | retrieval-only benchmark, no ROUGE | n/a | UNVERIFIED |

Not found: CAIL, COLIEE, Korean legal QA, CUAD-QA ROUGE-L numbers (COLIEE/ALQAC are entailment/retrieval, no ROUGE).
Typical "good": 0.3-0.35 fine-tuned small seq2seq on free-form human references; 0.5-0.6 strong LLM when references are statute-grounded; 0.65 when refs are LLM-generated templated style + SFT.

## (b) Extractive / copy baselines
- No paper found with explicit copy-article / lead-k ROUGE-L on legal QA. Closest: VLQA extractive sentence selectors 0.23-0.27 (below generative) with human-written refs.
- Reasoning (general, not sourced): copy-article ROUGE-L F1 is bounded by length mismatch (long article vs short answer -> low precision). Recall-only is high for copy. High (>0.7) occurs only when reference answers quote the article near-verbatim and length matches (best-sentence extraction). Templated refs ("Theo Điều X ... quy định ...") raise the floor for any system emitting the template.

## (c) Fine-tuning on reference style
- Style-matching SFT is the largest single lever: ViBidLAQA 0.657 (SFT on same-style LLM-generated data) vs VLQA zero-shot LLMs 0.38-0.60 and fine-tuned ViT5 0.32 on human refs. Hu et al.: +127% rel. ROUGE-L from SFT (UNVERIFIED). Gain comes from learning format/length/phrasing, not knowledge.
- Zero-shot LLM scores rise with model size but saturate near 0.6 even on VN statute QA.

## (d) Metric variants
- ROUGE-L F1 vs recall vs precision: recall-only (LegalBench-RAG ext.) or beta=1.2 (Hybrid RAG) inflates long outputs. Always report F1 (beta=1) with same variant for comparison.
- ROUGE-Lsum: LCS per sentence on newline splits; = ROUGE-L if no newlines (general knowledge, UNVERIFIED here). Usually >= ROUGE-L on multi-sentence text.
- Vietnamese: whitespace = syllable-level; word-level needs segmenter (VnCoreNLP, RDRsegmenter, underthesea ~98% seg F1) and changes scores. BARTpho paper notes ROUGE sensitive to detokenization https://arxiv.org/pdf/2110.04257 (UNVERIFIED). Char-level LCS inflates (shared letters/diacritics). Lowercase/punct stripping and diacritic handling shift numbers by points.
- Chinese/Japanese: rouge needs tokenizer (mLongT5 uses SPM) https://arxiv.org/pdf/2305.11129 (UNVERIFIED).

## Verdict
0.8-0.9 is not credible for free-form legal QA with independent refs (SOTA 0.3-0.65). Credible only if refs are templated/near-verbatim statute text AND length matched AND scorer lenient (recall, char/syllable level, normalised). Check own metric config before trusting.
