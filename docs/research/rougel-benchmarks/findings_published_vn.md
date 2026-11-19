# Published Vietnamese legal QA/RAG ROUGE-L (search 2026-11-20, 5 searches + 4 fetches)

Key answer: NO verified Vietnamese legal QA/RAG system reports ROUGE-L 0.8-0.9 on free-form answers with independent references.
Only >=0.8 values found: (a) VLegal-Bench task 2.3 "Legal Graph Structuring" 0.808 (not QA; structured output, zero-shot);
(b) SciTePress 2025 RAG: ROUGE-1 0.834 (R-L 0.756), refs poorly described, GPT-4 + template-generated QA. Both are not comparable to open-ended legal QA.

## Verified in source (WebFetch read)
| Source | Dataset | Metric | Value | Conditions |
|---|---|---|---|---|
| VLegal-Bench, arXiv 2512.14554 (https://arxiv.org/html/2512.14554v4) | 4.1 Summarization (384) | "ROUGE-L" (variant/tokenization unspecified) | best 0.3272 (gpt-4o-mini, gemma-2-9b) | zero-shot only |
| same | 4.2 Judicial reasoning (299) | ROUGE-L | best 0.4213 (CMC-AI-Legal-32B); gpt-4o 0.4017 | zero-shot |
| same | 4.3 Legal opinion (498) | ROUGE-L | best 0.4975 (gpt-4o); Qwen2.5-72B 0.4825 | zero-shot |
| same | 2.3 Legal graph structuring (296) | ROUGE-L | 0.808 (Claude Sonnet 4.5, Qwen2.5-72B) | structured/extractive-like output, not QA |
| Hien/Nhi/Huyen, SciTePress 2025 (https://www.scitepress.org/publishedPapers/2025/137512/pdf/index.html) | 45k legal docs, 2,000 QA pairs, 12 categories | ROUGE-1 / ROUGE-L (F/P/R unspecified; no R-2) | R-1 0.834, R-L 0.756; baselines R-1: std RAG 0.687, Legal BERT-QA 0.723, hybrid IR 0.695, trad. search 0.542 | GPT-4 generator, Milvus, RAGAs; refs "expert-validated" + partly template-generated; no ref-construction detail. Treat as weak/unreliable |
| ViBidLAQA_base (HF card https://huggingface.co/ntphuc149/ViBidLAQA_base) -- via search snippet, card not fetched: UNVERIFIED | Bidding-law abstractive QA | ROUGE-L 65.72, BLEU-4 39.54, BERTScore 86.65 | ViT5-base fine-tuned on same-domain refs; single narrow domain |

VLegal-Bench caveats: text/table inconsistencies; no RAG/agent ROUGE-L reported; last ~33k chars of page not read.

## Searched, no ROUGE-L found / not read
- ALQAC 2023/2024/2025 (https://sites.google.com/view/ALQAC-2025, ALQAC-2024, ALQAC2023): task 1 retrieval scored F2; QA task scored by accuracy/F1 style, not ROUGE-L (UNVERIFIED, from snippets). NeCo@ALQAC2023 (https://arxiv.org/pdf/2309.05500): extractive answer from retrieved article; no ROUGE-L seen.
- Zalo AI Challenge legal text retrieval: retrieval metric (F2), no ROUGE. Not searched specifically (UNVERIFIED).
- VLSP 2025 legal (https://preview.aclanthology.org/anlp-naacl/2025.vlsp-1.23.pdf): Qwen3-4B on MCQ/NLI/free-text QA; numbers not read.
- ViLegalLM (ACL Findings 2026, https://aclanthology.org/2026.findings-acl.1801/; code https://github.com/ntphuc149/ViLegalLM): abstract covers TF/MCQA/NLI + retrieval; no ROUGE-L visible. PDF saved but compressed/unread (/Users/hoangphu/.claude/projects/-Users-hoangphu-law-rag/c39eab88-793f-46a0-9917-1d7ae6fb83b2/tool-results/webfetch-1795107850722-mzpivy.pdf). ViLegalQwen = 1.5B/1.7B base models.
- ViLegalBench: only found VLegal-Bench (above); no separate "ViLegalBench" ROUGE source.
- URA-Law / URA-LLaMa (https://arxiv.org/html/2403.02715v2): general Vietnamese LLM eval, no legal ROUGE-L number found; no URA-Law paper found.
- ViLQA: nothing found.
- AITeamVN Vietnamese_Embedding (https://huggingface.co/AITeamVN/Vietnamese_Embedding), Vi-Qwen2-3B-RAG (https://huggingface.co/AITeamVN/Vi-Qwen2-3B-RAG/blob/main/README.md): retrieval metrics (legal domain) only; card benchmark table is for 7B; no ROUGE-L found.

## Conclusion for our work
ROUGE-L 0.8-0.9 is plausible only with extractive/copy-style answers whose references are derived from the retrieved article, or after fine-tuning on same-style references. Independent-reference generative zero-shot legal QA tops out ~0.33-0.50 (VLegal-Bench); fine-tuned same-domain ~0.66 (ViBidLAQA, unverified). If our dataset's reference answers are article-derived text, high R-L is a copy effect. Gaps worth a follow-up: ALQAC-2025 overview paper, VLSP 2025 paper table, ViLegalLM PDF tables (extract with pdftotext).
