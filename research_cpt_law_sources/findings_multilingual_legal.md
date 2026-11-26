# Multilingual / cross-lingual legal corpora for Vietnamese CPT (verified 2026-11-25)

Bottom line: NO well-known multilingual legal corpus contains meaningful Vietnamese. Vietnamese legal text must come from Vietnamese sources (see sibling findings in this folder). Only marginal/general-domain vi-en parallel data is strictly open-ish, and none is legal-domain.

| Source | Vietnamese? | Licence as written (HF API) | Gated | Size | Verdict / download |
|---|---|---|---|---|---|
| MultiLegalPile (joelniklaus/Multi_Legal_Pile) | NO. 24 langs, all EU (bg..sv, ga, mt). No vi | cc-by-nc-sa-4.0 (NC, subsets vary) | no | 689 GB | Exclude: no vi, NC |
| Multi_Legal_Pile_Commercial | NO (same 24 EU langs) | cc-by-sa-4.0 | no | subset of above | Exclude: no vi |
| Pile of Law (pile-of-law/pile-of-law) | NO. English only (US/Canada/intl) | cc-by-nc-sa-4.0 | no | ~256 GB | Exclude: no vi, NC. English legal only; marginal for cross-lingual transfer |
| MultiEURLEX (coastalcph/multi_eurlex; nlpaueb/multi_eurlex) | NO. 23 EU langs, no vi | cc-by-4.0 / cc-by-sa-4.0 | no | 65k docs | Exclude: no vi |
| UN Parallel Corpus (Helsinki-NLP/un_pc) | NO. ar,en,es,fr,ru,zh only | "other" (UN terms) | no | ~ millions of sent. | Exclude: no vi, unclear licence |
| OPUS-100 (Helsinki-NLP/opus-100, config en-vi) | YES, but general domain (mixed OPUS: subtitles, GNOME, etc.), little/no legal | "unknown" | no | en-vi up to 1M pairs | Doubtful licence -> not strictly open; skip |
| OPUS Wikipedia (Helsinki-NLP/opus_wikipedia, en-vi) | YES, general (not legal) | "unknown" (Wikipedia is CC BY-SA, but card unknown) | no | small | General vi only; not legal |
| Tatoeba-MT (Helsinki-NLP/tatoeba_mt) | YES, short sentences, not legal | cc-by-2.0 | auto/no | tiny | Negligible legal value |
| OPUS WIPO / EUbookshop / EMEA (opus.nlpl.eu) | No vi in EUbookshop (EU langs). WIPO/UN-style sets: not found with vi (unverified) | OPUS per-corpus, mostly unclear | n/a | n/a | Unverified; no vi evidence found |
| drixo/law-en-vi-es (HF) | Title suggests en-vi-es law; repo contains only .gitattributes (empty), no card | none | no | 0 | Unusable |
| Vietnamese-only legal sets on HF (YuITC/Vietnamese-Legal-Documents, license mit; 52100303-TranPhuocSang/vietnamese-legal-corpus-20k-raw, mit; vohuutridung/vietnamese_legal_dataset, apache-2.0) | YES, 100% vi | MIT / Apache-2.0 as tagged | no | small-mid (not size-verified) | Candidates, BUT licence tag is on a scrape of state-published laws (VN laws are public-domain-ish per Art. 15 Law on Copyright, but uploader's MIT is not authoritative). Check provenance; HIGH overlap risk with exam (same laws). Covered by sibling research |
| LegalBench / Legal terminology glossaries | LegalBench is English-only (task sets; per-task licences mixed, some CC-BY, some NC); no Vietnamese glossary with open licence found | n/a | n/a | n/a | Exclude |

Searches done: 1 web search (MultiLegalPile; confirmed no vi; paper says CC BY-NC-SA 4.0 overall, 24 EU langs) + HF API checks via curl. Not exhaustively searched: OPUS site per-corpus language lists (opus.nlpl.eu), Vietnamese govt bilingual (en-vi) legal translations (e.g. moj.gov.vn / vbpl.vn English translations) - these exist but licence/scrape rights unverified and not available as clean datasets.

## Notes
- Vietnamese content in all "multilingual legal" piles = zero. Do not spend CPT budget there.
- Doubts: OPUS "unknown" licences fail strict-open rule. HF-tagged MIT/Apache on scraped VN law dumps: tag is uploader's, verify provenance.
- Only marginal use: English legal (Pile of Law, NC -> excluded) for cross-lingual transfer; not allowed under strict-open.

## Contamination / overlap with Vietnamese legal QA exam
- Exam questions/answers cite specific laws; any Vietnamese statute corpus (and en-vi translations of same laws) contains gold articles verbatim -> leakage risk is high for law-text CPT, low for generic multilingual data.
- Filters: (1) 13-gram (token or char-normalized, lowercase, strip diacritics variant too) overlap vs dev/test questions AND reference answers; drop/mask docs/paragraphs with any hit. (2) Exclude gold articles: build set of (law id, article no.) cited in dev/test; drop those chunks, also amendments/consolidated versions and en translations of them. (3) Optional fuzzy: MinHash/embedding sim > ~0.9 vs questions. (4) Log removed counts; keep the filter list under version control. Decide per project policy whether to also drop whole cited laws (stricter) vs only cited articles.

## Sources
- https://huggingface.co/api/datasets/joelniklaus/Multi_Legal_Pile
- https://huggingface.co/api/datasets/joelniklaus/Multi_Legal_Pile_Commercial
- https://huggingface.co/api/datasets/pile-of-law/pile-of-law
- https://huggingface.co/api/datasets/coastalcph/multi_eurlex
- https://huggingface.co/api/datasets/Helsinki-NLP/un_pc
- https://huggingface.co/api/datasets/Helsinki-NLP/opus-100
- https://huggingface.co/api/datasets/Helsinki-NLP/opus_wikipedia
- https://huggingface.co/api/datasets/Helsinki-NLP/tatoeba_mt
- https://huggingface.co/api/datasets/drixo/law-en-vi-es
- https://huggingface.co/api/datasets/YuITC/Vietnamese-Legal-Documents
- https://arxiv.org/abs/2306.02069 (MultiLegalPile paper)
