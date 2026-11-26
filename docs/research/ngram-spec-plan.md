# N-gram (prompt-lookup) speculative decoding for the QA endpoint — plan (user go 2026-11-26: run AFTER the final run)
Why: answers quote the retrieved text almost verbatim, so the next tokens are usually already in the prompt; vLLM 0.30.0 on the box supports `--speculative-config` (method ngram). Lossless: greedy output must stay identical.
When: after both final chains have finished their test evals (GPUs free). Model: `data/models/FA_final` (chain A, the headline).
Steps:
1. Output identity: serve FA_final with and without `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'`, run 50 dev questions greedy through the API; the answers must be identical (or differ only in the last-token numerics) — otherwise stop.
2. Speed: `scripts/serve/vllm_flag_sweep.sh 16,32 30` with extra lines `ngram5` (above), `ngram8` (num_speculative_tokens 8, prompt_lookup_max 5), `ngram3` (3, 3) on layout A (one card); report p95, req/s, mean decode tokens/s vs `base`.
3. If p95 at 16/32 users improves ≥15% with identical answers: re-run the two-card capacity sweep (layouts B, C) with the best setting and update Section "Capacity under concurrent load" and docs/ops/docker.md (LLM_EXTRA). Also re-measure the final model with the base settings (capacity numbers in the report are for dpo_a2).
4. Log in docs/CHANGELOG.md (Step 38) and the report; if no gain, add the negative result to the deployment section.
Note: batching with many concurrent users reduces the benefit of speculation (decode becomes compute-bound); report both low and high load.
