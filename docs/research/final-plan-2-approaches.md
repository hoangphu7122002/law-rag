# Final plan: two approaches (one-shot 2B and ReAct 2B agent) - agreed 2026-11-22, runs 2026-11-23

Details: teacher-data-plan.md (data, agent, web, schedule), tail-mid-plan.md (diagnostics and history). Both approaches use the same retrieval, KG, store; the answering model is a 2B (< 4B). Test is touched once per approach at the end.

## 0. Common
- Data scope: train 2245 + dev 300 (never test, no test text to any external service). Train-only models are evaluated on dev300 greedy (paired bootstrap); final models train on train+dev.
- Teacher DeepSeek-V4 flash (Vibe), rollouts Qwen3.5-9B base; budget cap 60 M tokens / $40; licences of DeepSeek V4 and Vibe to be read and disclosed.
- Index: full68 (corpus + web_new_laws) vs control 4ac2a75c, compare Dieu@5 dev/test in the morning, use the better one.
- Guards: RAM >= 14 GB, disk >= 20 GB, GPU < 85 C, one job per GPU, util/temp check every ~10 min.

## 1. Preparation (CPU, tonight; GPU0 idle)
1. Measure on 2245 train questions: share naming a law number / Dieu / flag; precision of the "context lacks gold" signal (first_bad_step).
2. TDD: src/generate/profile.py (question profile regex + small-LLM fallback with JSON validation; retrieval profile + sufficiency signals), trajectory monitor script, sft_data options (--shuffle, inject ALL gold, --absent), web path deny-list.
3. Diagnostics: ROUGE by gold count and tail / mid / high before vs after DPO; 21 wrong-law dev cases vs KG families and dates; gold parser fix for laws named by title.
4. Read licences; fetch original documents for known gaps (TB 6150, TT 10/2025 appendix, draft of TT 39/2016; date <= end of 2025) into the local web store.

## 2. Data generation (GPU0 + API)
- Pass 1 (no tools): 9B answers from profile + top-5 (template opener). Grade (ROUGE-L >= .6 and cites gold Dieu). Passing rows = "context is enough" rows.
- Pass 2 (tools, only failing questions where the gold never reached the model): search_law (+question query), get_article, kg_related, web_search (guards: host allow-list, advice-page deny-list, 13-gram leak filter vs all references, post-2025 flag, TinyFish 500/h + cache + local store). Adaptive rounds: K=2 -> K=4 top-10 T .9 -> gold hint. Monitor every ~100 questions (completed, tool-call rate, gold seen, good, loops, tokens); coverage >= 60 % per split and per gold-count group; stop and fix if 9B unstable.
- DeepSeek repair from the first bad step (batches of ~20, strict JSON, tool re-run returns gold, copy 13-gram <= .5, template).
- Synthetic (30-40 % of rows, <= 50 %): articles excluding dev/test gold; mix single 40 / multi-article 20 / multi-law 10 / version family 10 / notice-appendix-draft 5 / vague paraphrase 10 / no-answer 5-10; hard contexts from real retriever mistakes (sibling Dieu, other version first, similar law, shuffle, k in [3,10], gold dropped -> exact "lacks gold" label); style, near-duplicate and per-law cap controls.

## 3. Approach 1 - one-shot 2B (GPU0 lane)
- Configs (2 seeds each, no early stop): A current v3b (3 existing seeds .6325 / .603 / .6134), B = + shuffle (~50 %) + inject-all gold, C = B + 15 % absent, D = C + teacher and synthetic rows.
- Then DPO from the best SFT (on-policy, as dpo_a2 recipe), final SFT on train+dev, test once (greedy + MBR8).

## 4. Approach 2 - ReAct 2B agent (GPU1 lane)
- Architecture: question profile -> retrieval (dense top-50 -> rerank top-5) -> retrieval profile + sufficiency signals -> 2B agent (<= 4 turns, answer directly or call a tool) -> tools search_law / get_article / kg_related / web_search -> template answer -> evidence gate + citation check.
- Rows: (a) one-shot rows in agent format with the tricks, (b) tool trajectories S1 / S2 / synthetic dropped-gold recoveries, (c) "context is enough, do not search", (d) the question-profile extraction step. E = SFT (2 seeds); F = DPO / multi-turn GRPO if E improves; final on train+dev; test once.
- Also report: agent with a big model (DeepSeek / GLM) and 9B on dev (ceiling), closed store vs web-enabled lines, tool-call rate, turns, latency, ROUGE split by gold in the initial top-5.

## 5. Timeline 2026-11-23 (estimates from the compute log)
| Slot | GPU0 (one-shot) | GPU1 (agent / spare) | CPU / API |
|---|---|---|---|
| Tonight | pass-1 rollouts if profile is ready, else idle | finish full68 (dev, test) | section 1 |
| 08:00-10:30 | SFT B s1, then C s1 | SFT B s2, then C s2 | grading, DeepSeek repair, synthetic generation, build rows |
| 10:30-13:30 | SFT D s1, s2 | SFT E s1, s2 | bootstrap B / C vs A, monitor |
| 13:30-16:30 | DPO from best one-shot, final SFT train+dev | F if E improves (else spare seed), final agent train+dev | report text, tables |
| 16:30-17:30 | test once (greedy, MBR8) | test once (agent) | CHANGELOG, compute-log, PDF |
Gaps: third seed for any pair whose paired difference is within .005-.015.

## 6. Success criteria and reporting
- A config counts only if its 2-seed mean beats A by more than the noise (SFT sd .015) with a paired bootstrap on dev300; otherwise reported as exploratory / null (honest negatives stay in the report).
- Expected effects are small (tools about +.03, diversity +.01-.02, doubling of data +.016); the old ReAct 2B was .537 / .574 vs one-shot .633.
- Both approaches reported side by side; the graded system stays the best one-shot unless the agent wins on dev and test.

## 7. Open / deferred
Budget cap confirmation; DeepSeek V4 and Vibe licences; whether web at answer time is allowed by the assignment (extra line only); API validity status field (later); chunking Stage 2 and sem1000 (no variant passed the gate); fill author names, nllp26 authors, rotate the pasted keys (Mistral, TinyFish, HF write token).
