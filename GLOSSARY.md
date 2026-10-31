# GLOSSARY

| Term | Meaning |
|---|---|
| corpus | the 16,441 law text files in `Dataset.zip/export_1` |
| RES | the 2,745 question–answer pairs (course sheet), saved as `data/raw/res.csv` |
| law id (số hiệu VBPL) | a document number such as `50/2024/TT-NHNN`; also the corpus file name with `/`→`_` |
| article (Điều) / clause (khoản) / point (điểm) | the legal units inside a document; the natural chunk boundaries |
| split | fixed seeded partition of RES into `train` / `dev` / `test` (`data/processed/split.json`) |
| baseline | the course notebook pipeline (sbert + FAISS k=5 + Qwen1.5-1.8B), re-run on our `test` split |
| hit@k (law) | share of questions whose cited law id appears among the top-k retrieved chunks |
| experiment | one YAML in `config/experiments/` → one run directory `results/runs/<id>/` |
