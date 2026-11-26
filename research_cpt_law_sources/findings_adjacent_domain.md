# Adjacent-domain / extra Vietnamese law sources (subagent sonnet55-med-adjacent-domain; HF card JSON checked, sizes/tokens estimated at 3.4 chars/token)
| id | licence (as written) | gated | size | est tokens | verdict |
|---|---|---|---|---|---|
| th1nhng0/vietnamese-legal-documents | cc-by-4.0 (texts public domain, Law 104/2016 & 64/2025) | no | content ~172k docs; legacy_content 518k docs 10.7 GB raw HTML (3.5 GB parquet) | legacy ~2-3B, new ~1B (guess) | BEST: vbpl.vn official portal, different site from TVPL; strip HTML, dedup on so_ky_hieu vs tmquan |
| vohuutridung/vietnamese-legal-documents | cc-by-4.0 | no | 518k docs, 3.6 GB | ~2-3B | scraped from thuvienphapluat -> heavy overlap with tmquan; only if dedup finds new docs |
| duyet/vietnamese-legal-instruct | cc-by-4.0 | no | 467k pairs | ~0.4B | synthetic from th1nhng0, generator licence unknown: skip |
| wikimedia/wikisource 20231201.vi | cc-by-sa-3.0 + gfdl | no | 5.5k docs 48 MB | ~12M | tiny; filter Hiến pháp/Luật/Bộ luật/Nghị định |
| bigscience-data/roots_vi_wikibooks | cc-by-sa-3.0 | gated auto | 9 MB | ~6M | low priority |
| FineWeb-2 vie_Latn URL-filtered (gov domains: chinhphu.vn, baochinhphu.vn, *.gov.vn, vbpl.vn) | odc-by | no | 130 GB scan | 50-300M guess | OK, needs scan; commercial legal sites' commentary stays under owners' copyright |
Rejected (licence not stated / other / empty / unknown provenance): YuITC/Vietnamese-Legal-Documents (MIT but raw = BKAI-Legal-Retrieval, licence not stated), NamSyntax, AlexNgV, anhnon, tardis2, ThanhVu101/*, GreenNode/Vietnamese-Law-hn-mined-v2, kiil-lab, dinhquangson, Monmoonluna, Hiepppp, thangvip/combined-*, Tucker6742/Vietnamese-government-document (apache header, empty README, 24 GB), trungbb8 news, many clones of th1nhng0.
Notes: Vietnam IP Law Art. 15(2) excludes legal/administrative documents from copyright (law text low risk wherever hosted; commentary/Q&A/news not). No open HF set found for admin procedures, legal news, legal consultation, banking/tax/customs.
Sources: https://huggingface.co/api/datasets/<id> for each id above; https://commons.wikimedia.org/wiki/Commons:Copyright_rules_by_territory/Vietnam ; https://www.wipo.int/wipolex/en/text/449011
