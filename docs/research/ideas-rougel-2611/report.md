# Ideas to raise ROUGE-L (web research 2026-11-26; numbers from search snippets, unverified)
1. Quote-then-answer + substring-constrained / snapped decoding — Pleias-RAG 2504.18225 (350M/1B trained to quote literally), CopyGen (NeurIPS 2026 poster; training-free copy-constrained decoding controlling span extent and stop), FullCite 2606.07130, GENRE trie decoding. Check oracle-span ROUGE-L bound first (we have scripts/analysis/extractive_oracle.py).
2. Best-of-n rejection-sampling SFT (RFT): from the n=8 samples keep top ROUGE-L within a length band; BOND 2407.14622, best-of-teacher distillation 2605.09725. No data cost.
3. Length-matched, ROUGE-ranked pairs + NLL term on chosen (LD-DPO, R-DPO 2403, SimPO 2405.14734, LMPO 2502.14643, MWPO). Our bias is DPO→shorter (opposite sign), so match lengths rather than penalise.
4. GRPO with ROUGE-L F1 reward + mild length term (length-collapse caveat: 2510.19325, 2504.05185). Fallback only.
5. Clause-level select-then-generate (DaR, PARSER/KoBLEX 2509.01324, VLSP DRiLL cross-encoder rerank with clause segmentation); no statute-specific select-then-generate paper found → open idea. Our lexical khoản selector: top-1 47%, top-3 82%.
6. Steering vectors for length/format (Stolfo et al., ICLR 2025, works on Gemma-2 2B); 2605.23040 warns of diluted style steering in small models. Covered by the weight-space agent.
