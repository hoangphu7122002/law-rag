# configs/queues/final/ (9 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/final/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `final_chainA_g0.txt` | FINAL RUN chain A (GPU0, seed 1): PiSSA SFT on train+dev -> DPO round 1 (standard pairs) -> DPO round 2 (mixed extent pairs) -> test ONCE (greedy, then MBR8). Pre-declared in docs/research/final-run-plan.md. |
| `final_chainB_g1.txt` | FINAL RUN chain B (GPU1, seed 2): PiSSA SFT on train+dev -> DPO round 1 (standard pairs) -> DPO round 2 (mixed extent pairs) -> test ONCE (greedy, then MBR8). Pre-declared in docs/research/final-run-plan.md. |
| `final_data.txt` | FINAL RUN step 0 (user go 2026-11-26): the q35_C recipe on train+dev (data q35_CF). Same flags as q35_C plus --splits train,dev. Test stays untouched. |
| `q4b_g1.txt` | 4B scaling point (user 2026-11-27: "train đầy đủ hết"): the reported 2B recipe (dpo_a2 = SFT v3b -> on-policy DPO r1 -> r2) re-run unchanged on Qwen/Qwen3.5-4B, GPU1. |
| `q9b_g0.txt` | 9B scaling point (user 2026-11-27: "train đầy đủ hết"): the reported 2B recipe (dpo_a2 = SFT v3b -> on-policy DPO r1 -> r2) re-run unchanged on Qwen/Qwen3.5-9B, GPU0. |
| `think2_g0_release.txt` | After the rationalise-only thinking data is written, stop the 9B on GPU0 so GPU0 can go back to Triton. |
| `think2_g1.txt` | Thinking SFT, take 2 (Nov 27). Take 1 kept 1/1683 rows: blind 9B thoughts are English, run past max-tokens; the rationale filter also rejected every rationale that ruled out |
| `think2_nothink_g1.txt` | Diagnostic: thinking SFT model (q35_T_pissa) with thinking OFF at inference, dev. Separates "fewer/other training rows" from "thinking at inference" (thinking on: dev .5852 vs V_pissa .6414). |
| `think_g1.txt` | Thinking-mode SFT with GENUINE reasoning (user: needs proper thinking): Qwen3.5-9B (thinking on) answers each q35_C prompt blind, n=4; a sample is kept when its final answer cites exactly the reference's Điều and overlaps it (ROUGE-L>=.5); training row = prompt opening <think> + the kept thought + </think> + the REFERENCE answer. Then SFT PiSSA r32 3 epochs on q35_T and a dev eval with thinking on. Compare with V_pissa (non-thinking SFT, dev .6414). GPU1, after the spec-decoding bench. |
