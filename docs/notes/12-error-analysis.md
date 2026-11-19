# Error analysis — SFT Qwen3.5-2B, dev (2026-11-19)

Run: A4_q35_off (v0, k5, rougeL .6026) and A5_v3b (v3b, shots 2, k5, rougeL .6325).

| Group (v0) | n | rougeL |
|---|---|---|
| gold Điều in top5 and cited | 205 | .702 |
| gold in top5, cites another Điều | 28 | .397 |
| retrieval miss | 25 | .295 |

- Retrieval/Điều errors cost ≈ .06 rougeL if fully fixed (estimate); the rest is a ceiling of lexical metrics vs paraphrased references.
- Train contexts lacked the gold Điều in 14 % of rows (gold in context 85.8 %) → v2 (dynamic k + gold injection): no dev gain alone.
- Refusal template scores rougeL .066 vs guess .354 → refusal rows limited (8–20 %), built from unrelated contexts. Refusal rate in v3b predictions 2/300.
- v3b gain comes mostly from few-shot at eval: shots 0 → .5937, shots 2 → .6325 (k10: .6122). v3 shots2 .6266. hp A/B (alpha64, d0, lr2e-4) .5988 ≈ v0.
- Length (v3b): pred median 315 vs ref 294 words; truncating to ref length would give .660 (+.028). Short refs (<200 words) are over-answered 1.8×–3×.
- Reranker rr_v0 lifts art@1 .587 → .712 but did not lift generation with v0; re-testing with v3b (queue-1119-093835).
- Caveat: 151/188 dev answers share a 13-gram with train answers → dev gain partly style; judge on test only.

Next: length control (answer-length hint / shots chosen by length), Haiku-generated QA per Điều (decontaminated), Điều-selection training.

## RL fallback (user, 2026-11-19)
If SFT plateaus → GRPO on the SFT checkpoint (TRL GRPOTrainer + colocated vLLM, see 04-research-post-training-rl.md).
- datamate has NO RL (SFT datagen only; its "reward" mention is the PA metric rewarding abstention) → nothing to copy; its datagen rules (contrast rows, decontam) already reused.
- RL data = prompts (question + retrieved context + shots) with a verifiable reward vs a reference: course train 1452 (decontam) have references → no generation needed. More prompts: Zalo 3k questions (gold article known; reference = Haiku-written template answer, or reward on overlap with the gold article), synthetic questions per Điều via Haiku.
- Reward = .5 rougeL + .3 cosine + .2 citation-in-context; penalties: copy ratio, length vs ref. Refusal gets low reward (template rougeL .066).
- Haiku pacing: small batches (≈20 items/call), sequential, pauses between calls; no bursts.
