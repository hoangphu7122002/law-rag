# Report: extra open-licence law text for CPT (target ~200M tokens total)
Findings files: findings_legal_corpora.md, findings_adjacent_domain.md, findings_multilingual_legal.md.
- Multilingual legal corpora (MultiLegalPile, Pile of Law, MultiEURLEX, UN, LegalBench): no usable Vietnamese; Pile of Law / MultiLegalPile are CC-BY-NC-SA -> not used.
- Usable new law text (all cc-by-4.0, ungated): th1nhng0/vietnamese-legal-documents (vbpl.vn laws, ~2-3B tokens), tmquan/phapdien-moj-gov-vn (~41M), tmquan/cbba-toaan-gov-vn court judgments (~5B), tmquan/anle-toaan-gov-vn (~15M), wikisource vi (~12M), FineWeb-2 vie gov-domain subset (50-300M, scan needed).
- Overlap: vbpl.vn vs thuvienphapluat sets (dedup on so_ky_hieu). Judgments cite laws, may contain exam-related facts -> 13-gram filter vs dev/test questions and references; PII filter.
