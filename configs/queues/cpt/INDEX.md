# configs/queues/cpt/ (5 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/cpt/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `cptC_g0.txt` | SFT FROM the CPT v2 model with PiSSA on data q35_C (q35_cptC_pissa); best-val and last-epoch dev+test. Waits for cpt_report.json. Compare with V_pissa (dev .6414 / test .6307) = effect of CPT; C vs D3 on the same CPT base = effect |
| `cptD3_g1.txt` | SFT FROM the CPT v2 model with PiSSA on data D3 (effect of data vs cptC_pissa);  (/home/app/phunguyen/law_rag/data/models/q35_cpt_v2), best-val and last-epoch dev+test; waits for the CPT run to write cpt_report.json. Recipe = vari |
| `cptD3_g1_resume.txt` | cptD3_pissa evals resume (the first queue stopped at the 2nd serve: GPU1 memory not yet released). test best-val, dev+test last. |
| `cpt_articles.txt` | CPT augmentation input: one jsonl line per law article (CPU, loads the docstore; same laws.jsonl rebuilt deterministically). |
| `cpt_data.txt` | CPT data: text of the laws cited by the train/dev/test questions, from the index nodes, packed into 2048-token sequences (CPU, loads the docstore). |
