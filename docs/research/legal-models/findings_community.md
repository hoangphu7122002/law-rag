# Community Vietnamese legal models <4B — primary-page facts (agent report 2026-11-20; "primary" = WebFetch summary of the HF page, not raw card text)

| Model | Params / base | Method / data | Instruct? | License | Eval reported |
|---|---|---|---|---|---|
| bqbbao6/Qwen2.5-3B-legal-vn | 3B, Qwen2.5-3B-Instruct (unsloth bnb-4bit build) | QLoRA r16 a32 (Unsloth + TRL); data not stated | yes | Apache-2.0 | none; unclear full weights vs adapter |
| thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2 | ~1.7B, Qwen3-1.7B-Base lineage (card contradicts itself on the phase-1 base) | GRPO (TRL) ~1500 steps on hard_level_qa.jsonl (syllogistic QA, size not given); reward incl. citation 15 % + hallucination 10 % | reasoning style | Apache-2.0 | none |
| VLSP2025-LegalSML/qwen3-1.7b-legal-pretrain | 1.7B, Qwen3-1.7B | full-parameter continued pretraining on ~144k texts (~96k legal docs + ~48k legal news) | base only | research-only | none for the 1.7B. Paper (aclanthology.org/2025.vlsp-1.22.pdf, Bosch@AI_Team) fine-tunes the 4B: LoRA r32 on 200K GPT-generated QA; best 89.73 QA / 96.00 NLI (thinking mode, public test), private avg 81.08 (vi-law-qa 92.67, NLI 97.00, syllo 53.58); metric for free text not named |
| ntkhoi/Llama3.2-3B-Vietnamese-Law | – | page returned HTTP 401 | – | – | UNVERIFIED |

None reports ROUGE-L / BLEU on legal QA generation, so no number is comparable to ours (dev ROUGE-L ~.65).
