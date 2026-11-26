# src/train/cpt/

continued pretraining: corpus build, cleaning, augmentation, mixing, closed-book probe, 2-GPU trainer.

Import as `train.cpt.<module>`; run as `python -m train.cpt.<module>` or `python src/train/cpt/<module>.py` with `PYTHONPATH=src`.

| module | what |
|---|---|
| `cpt.py` | Continued pretraining of the answer model on the law text (docs/research: 'pretrain Qwen3.5-2B on the laws our questions are about'). |
| `cpt_clean.py` | Cleaning and curation of external CPT text (docs/research/cpt: quality + dedup + PII + no eval leakage matter more than raw volume). |
| `cpt_data.py` | Continued-pretraining data: the text of the laws our questions are about, packed into fixed-length token sequences. |
| `cpt_mix.py` | Mix the CPT sources into one packed training file (docs/research: paraphrase/QA/summary forms of each fact, plus replay against forgetting). |
| `cpt_probe.py` | Closed-book cloze probe for the CPT go/no-go (docs/research): does the model complete a number of a law article without any context? |
| `cpt_synth.py` | CPT augmentation (docs/research: knowledge must be seen in several surface forms to be extractable): for every law article a teacher LLM |
