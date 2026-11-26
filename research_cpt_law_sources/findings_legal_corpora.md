# Direct Vietnamese legal corpora (subagent sonnet55-med-legal-corpora, HF API checked 2026-11-25; tokens ~3.4 chars/token, rough)
| id | licence | size | docs | est tokens | text field | note |
|---|---|---|---|---|---|---|
| tmquan/cbba-toaan-gov-vn (config documents) | cc-by-4.0 | 4.71 GB parquet (19.95 GB raw) | 1,542,247 court judgments | ~5B | markdown | partly anonymised, PII filter; 26.6% of crawl missing |
| tmquan/anle-toaan-gov-vn | cc-by-4.0 | 59 MB | 1,844 (9 án lệ + rulings) | ~15M | markdown | |
| tmquan/phapdien-moj-gov-vn | cc-by-4.0 | 166 MB | 65,967 articles (Bộ Pháp điển) | ~41M | content_text | official consolidated code; overlaps our laws |
| th1nhng0/vietnamese-legal-documents (legacy_content / content) | cc-by-4.0 | 10.7 GB / 5.6 GB raw | 518k / 171k | 2-3B / ~1B | content / content_html (strip HTML) | vbpl.vn; overlaps tvpl set, dedup on so_ky_hieu |
| anhquan12/vietnamese-legal-documents | cc-by-4.0 | 3.2 GB | 178k HTML | ~0.8B | content_html | subset of th1nhng0; mirrors redundant |
| 52100303-TranPhuocSang/vietnamese-legal-corpus-20k-raw | mit (front matter only) | 2.56 GB | 19k | 0.3-0.7B | full_text | provenance not stated (likely tvpl scrape) -> doubt |
| daitavan/Vietnam-Law-Raw-Data | apache-2.0 (front matter only) | 138.6 GB | ~290 files | unknown | raw doc/docx | provenance unknown, binary -> skip |
Other tmquan: thuvienphapluat-vn-hdpl/tnpl/htpl, pbgdpl-vn-legal-qna licence 'other' -> reject; dichvucong-gov-vn cc-by-4.0 27 MB admin procedures (not law text); nso/sapnhap cc-by-nc -> reject.
Open but not CPT-useful: duyet/vietnamese-legal-instruct, YuITC (retrieval QA), Monmoonluna (images, SA).
Caveat: cc-by labels are uploaders' relicensing of government-published text; attribution required.
Sources: https://huggingface.co/datasets/<id> for each id.
