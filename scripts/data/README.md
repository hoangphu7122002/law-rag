# scripts/data/

building data: CPT corpus / augmentation / external text, synthetic QA, SFT subsets, missing laws, manual-check sheets.

Run from the repo root: `PYTHONPATH=src python scripts/data/<file>.py ...` (scripts are not a package; their tests add this folder to `sys.path` in `tests/unit/scripts/conftest.py`).

| file | what |
|---|---|
| `cpt_build_v2.py` | Curated external CPT text (v2): open-licence legal sources + replay, cleaned, deduplicated, PII-masked, decontaminated against dev/test. |
| `cpt_external.py` | External CPT text from open-licence Hugging Face datasets (no gated / unlicensed sources), decontaminated against dev/test. |
| `cpt_synth.py` | CPT augmentation: paraphrases / forward+reverse QA / summaries per law article via DeepSeek (vibe_call.py). See src/train/cpt_synth.py. |
| `fetch_missing_laws.py` | Fetch the cited laws that the store lacks (data/processed/missing_laws.json, written by the coverage check) from luatvietnam.vn. |
| `manual_check_sheet.py` | A CSV for a human check of answers: 20 test answers, one from each ROUGE-L stratum, with the columns to fill in. |
| `subset_sft.py` | A nested, seeded subset of an SFT dataset by question (every row of a kept question stays), for the learning curve. |
| `synth_gen.py` | Synthetic Q/A generation with a teacher LLM (DeepSeek-V4 flash via vibe_call.py). See src/train/synth_gen.py. |
