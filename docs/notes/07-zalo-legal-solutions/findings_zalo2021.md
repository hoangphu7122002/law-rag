# Zalo AI Challenge 2021 - Legal Text Retrieval: solution findings

Date: 2026-11-18. Method: 6 web searches/fetches. Coverage is PARTIAL. The official
leaderboard / top-3 team list was NOT found (challenge.zalo.ai portal not reachable via search).
Everything below is limited to what was verified in public repos/papers. Unverified items are marked.

## 0. Key caveat
- Could not confirm who placed 1st/2nd/3rd, nor private-test F2 scores. [UNVERIFIED / NOT FOUND]
- Two public repos exist that target the Zalo data ("zac2021-ltr-data"); neither README states placement.
- Next step if needed: Zalo portal (challenge.zalo.ai), Vietnamese blogs (Viblo/Medium/Facebook ML groups),
  GitHub search "zalo ai challenge legal text retrieval", "zac2021-ltr".

## 1. Dataset format and stats (matches our task)
- Files in `zac2021-ltr-data/` (from repo READMEs): `train_question_answer.json`, `public_test_question.json`,
  plus legal corpus (`legal_corpus.json`; name from repo layout, not verified in README text).
- Task: given a question, return relevant article(s), each identified by (law_id, article_id).
  Exact JSON keys (question_id / text / relevant_articles[{law_id, article_id}]) NOT confirmed from a
  fetched page [UNVERIFIED; matches my memory of the challenge, check against our own data].
- Stats (NAACL 2025 short paper arXiv 2412.00657, citing Zalo AI Team 2021): 61,425 corpus passages
  (61,060 per table in another listing; difference unexplained), 3,196 questions total
  ("3196 questions and 61425 articles" from a search snippet), 2,556 train + 640 eval.
- Metric for the challenge: F2 (recall-weighted), macro-averaged over queries; variable number of
  returned articles per query => needs a score threshold / top-k cut. (F2 definition shown in arXiv 2209.14494;
  that it was the official Zalo metric is consistent but not independently confirmed.)
- ~95% of questions have 1-3 relevant articles (arXiv 2209.14494, on its own dataset variant).

## 2. Repo A: CuongNN218/zalo_ltr_2021 (strongest documented Zalo-era pipeline)
URL: https://github.com/CuongNN218/zalo_ltr_2021  (credits teammates Dzung Le, Hong Nguyen)
Placement/score: not stated in README [UNVERIFIED]. Linked paper below is by different authors
(Pham, Nguyen, Do) who cite this repo as "the solution for an annual AI competition".
- Data: only Zalo corpus + train QA (no extra crawl mentioned).
- Pipeline:
  1. BM25 -> top-20 / top-50 negatives, pair files.
  2. Continue MLM pre-training on the legal corpus (PhoBERT-large, ViBERT; base/ELECTRA variants listed).
  3. Condenser and Co-Condenser pre-training (code patched for newer transformers).
  4. Sentence-Transformer bi-encoders w/ contrastive loss, 4 configs; round 1 BM25 negatives,
     round 2 hard negatives mined from round-1 model.
  5. Ensemble of the four round-2 models: PhoBERT-large 0.1, Condenser 0.3, Co-Condenser 0.4, ViBERT 0.2.
  6. Post-processing: multiply cosine sim by BM25 score; keep articles within 2.6 of the top score,
     max 5 per question; fix typo "nd-cp".
- Segmentation: not stated in README (paper below used VnCoreNLP).
- Unusual: round-3 training and pseudo-labeling tried; pseudo-labeling helped single models but HURT the
  ensemble. No cross-encoder reranker. Ships pre-encoded embeddings (`encoded_legal_data.pkl`).

## 3. Repo B: phuongnm-bkhn/legal_text_retrieval
URL: https://github.com/phuongnm-bkhn/legal_text_retrieval  (MIT; cites JNLP COLIEE 2020/2021 papers)
Placement/score: none reported [UNVERIFIED].
- Step 1: training data with negatives chosen by mixed TF-IDF cosine + BM25 over top-150 articles.
- Step 2: fine-tune `vinai/phobert-base` (optionally NlpHUST model) as a pair classifier
  (COLIEE-JNLP style = cross-encoder style), 5 epochs in example.
- Tokenization: VnCoreNLP. If no test file given, holds out 33% of train QA as test.
- Looks like a COLIEE-style BM25/TF-IDF -> BERT classifier rerank; a different philosophy from repo A.

## 4. Paper tied to Repo A: "Multi-stage Information Retrieval for Vietnamese Legal Texts"
arXiv 2209.14494 (Pham, Nguyen, Do): https://arxiv.org/pdf/2209.14494v2.pdf
CAUTION: evaluated on the Kien et al. corpus (8,436 docs / 114,177 articles, QA references only 1,709
articles), NOT the Zalo corpus. Numbers are NOT Zalo leaderboard scores. Still the best ablation evidence.
- BM25+ (rank_bm25), stopwords removed, VnCoreNLP word segmentation.
- 3 training rounds: negatives k=35 (BM25+), then 20 (from round 1), then 15 (from round 2). batch 8,
  4 epochs, lr 1e-5, contrastive loss. Hardware: single P100 on Colab.
- MLM fine-tune 20 epochs on legal corpus, then Condenser (luyug/Condenser code) from PhoBERT.
- Article length: PhoBERT max 256 tokens, ViBERT 512; ~28% (syllable) / ~24% (word) of articles > 256.
- Results (Recall@20 / F2), single models: BM25+ 0.557/0.221; PhoBERT-base(word) 0.954/0.721;
  PhoBERT-large(word) 0.954/0.727; Condenser-PBL(word) 0.954/0.724; ViBERT(word) 0.921/0.669.
- Hybrid score: sqrt(BM25)*cos_sim best for F2 (PhoBERT-large word: R@20 0.967, F2 0.741);
  BM25*cos_sim best for recall (R@20 0.970). Linear alpha mixing rejected (score scales differ).
- Findings: word-segmented input better for PhoBERT-family; syllable better for ViBERT (matches pretraining);
  hybrid > dense alone; Condenser gives only marginal/no gain over plain PhoBERT here;
  F2 falls as number of gold articles per query grows (threshold too tight).
- Prior baselines on that data: Attentive CNN F2 0.477, XLM-R 0.201; ES+Attentive CNN R@20 0.825.

## 5. Other relevant evidence
- arXiv 2412.00657 (BKAI, NAACL 2025 short): https://arxiv.org/html/2412.00657v1
  Uses 80/20 split of Zalo 2021 train (2,556 train queries). Own eval on Legal Zalo 21:
  BM25 MRR@10 51.5 / R@100 72.3; vietnamese-bi-encoder 80.7 / 80.1; BGE-M3 64.4 / 75.3;
  mE5-base 59.7 / 84.1; their bi-encoder 79.3; ColBERT best MRR 84.2, R@100 80.8.
  Hard negatives mined with BGE-M3; passages chunked to 256 tokens; InfoNCE. Not challenge F2.
  Useful: shows dense general models weak zero-shot on Zalo; domain fine-tuning matters.
- ALQAC 2021/2022 (separate competition, similar format, F2 metric): top ALQAC21 F2 0.8807
  (arXiv 2204.10717); ALQAC22 1st place used BM25 candidates + fine-tuned RoBERTa pair classifier
  (arXiv 2211.02200). Different data; do not mix with Zalo numbers.
- 2025 VLSP DRiLL entries use hybrid BM25+dense first stage + BERT cross-encoder rerank
  (aclanthology 2025.vlsp-1.20); not Zalo.

## 6. Takeaways for our pipeline
- Recipe seen on Zalo-style data: BM25 -> bi-encoder (PhoBERT-large, word-segmented) w/ iterative
  hard-negative mining (BM25, then model negatives) -> multiply BM25 by cosine -> threshold relative to
  top score (F2 cut) -> ensemble of several encoders. Condenser/Co-Condenser used but gain unproven.
- Variable-length output: relative-to-top threshold (2.6 in repo A, tuned) is key for F2.
- Truncation: PhoBERT 256 limit hurts long articles; ViBERT 512 helps; consider chunking.
- Missing: confirmed top-3 teams, cross-encoder-heavy winners, extra-crawl usage. [NOT FOUND]
