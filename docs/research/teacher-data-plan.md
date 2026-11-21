# Teacher-data plan: 9B ReAct + DeepSeek correction -> 2B one-shot SFT (decided 2026-11-22, run from 2026-11-23, no pilot)

Decisions (user + defaults): teacher = DeepSeek-V4 flash via /vibe (alias deepseekflash); rollouts by base Qwen3.5-9B (HF hub, same family as the 2B; not the codeact checkpoints); scope = ALL train (2245) + dev (300), never test; student = a 2B CORE AGENT (user, 2026-11-22: train the 2B to be a good enough ReAct agent); the one-shot 2B (format B) stays as control and fallback, both < 4B; no web tool (leak risk).

## Stage 0 - preconditions (morning)
- Index full68 (corpus + web_new_laws) finished and compared with control 4ac2a75c (retr_compare); use the better one for tools and training context.
- Box guard: RAM >= 14 GB, disk >= 20 GB free, GPU temp < 85 C; embed :8101 + rerank :8201 on GPU1 up; one heavy job at a time.
- Licence (partial, record in the report): DeepSeek V4 weights reported MIT (Alibaba Cloud listing, secondary news); DeepSeek Open Platform ToS allows using outputs to train other models incl. distillation; V4 LICENSE file not read; the call goes through Mistral Vibe, whose terms were not read either. Read both before the full run; disclose in the report.
- Budget cap: total --max-price 40 USD for the teacher; ledger project label `tdata`; price of deepseekflash is not listed (ledger shows 0), so cost is tracked in tokens (stop at 60 M tokens total).

## Stage 1 - 9B ReAct rollouts (GPU0)
- vLLM Qwen3.5-9B bf16 on GPU0 (~18 GB), tools search_law / get_article / kg_related over the chosen index (embed/rerank on GPU1), max_turns 4, K = 2 rollouts per question at T 0.7, non-thinking.
- Output data/gen/tdata/rollouts_{train,dev}.jsonl, one line per rollout with trajectory, tool results, final answer.

## Stage 1b - trajectory monitoring and adaptive sampling (user: watch the 9B trajectories so they cover; sample more if the 9B is unstable)
- Initial context is GIVEN (user): the pipeline top-5 goes into the prompt (cfg agent_initial_k = -1, retrieve.k = 5) and the 9B may call tools for more; min_tool_calls = 0 (a question whose context suffices needs no search); search_with_question on for every search_law call. Round 2 gives top-10 as initial context. Optional datamate-style evidence gate (cfg `check`, one nudge) stays off in round 1.
- Monitor (script prints every ~100 questions and a final table, by split / gold count / tail-mid-high): completed rate (final answer, valid tool JSON), tool-call rate and mean calls, share of trajectories that saw a gold Dieu (react_data.first_bad_step), good rate (ROUGE-L >= .6 + cites gold), loop / empty-answer / over-length rate, tokens per question.
- Coverage = share of questions with at least one good trajectory (S1) or an accepted repair (S2); target >= 60 % per split and per gold-count group, no group below 40 %.
- Adaptive rounds: round 1 K=2 (T .7). Questions without a good trajectory -> round 2 K=4 (T .9, fresh seeds). Still none -> round 3 with the gold Dieu hinted in the prompt (self-distillation fallback, hint never copied into the stored row) -> then DeepSeek repair. A question gets at most these three rounds.
- 9B unstable (completed < 90 % or tool-call rate < 80 % or empty/loop > 5 %): stop, fix the system prompt / parser, restart that round; do not feed bad rollouts onward.

## Stage 1c - retrieval profile (like df.info(), user idea) given to the 9B at the start
Deterministic, no LLM, ~300-500 tokens, built from the retrieval result + chunk metadata + KG; replaces exploratory tool calls (list_articles / kg_related), the 9B calls get_article only to read text.
```
[HO SO TRUY XUAT]
Cau hoi: law number / Dieu mentioned? (so hieu, Dieu) ; type cue (dieu kien / tham quyen / thoi han / quyen-nghia vu)
Top luat (rerank order): 1. <so hieu> - <title> - <nam> - <hieu luc> - family: <thay the / sua boi ...>
   retrieved Dieu: 12 (<heading>), 15 (<heading>) ; the law has Dieu 1..37 (table of contents on demand)
Same-family laws in the result: <21/2019 <-> 29/2024> (neutral: no preferred version)
```
Neutral by design (no "use the newer law": refuted). Used only in the teacher rollouts; the 2B one-shot student never sees it, so there is no train/test mismatch. Earlier decision stays: no question rewriting / profiling in the scored system. Code: src/generate/profile.py build_profile(...) (TDD) + cfg flag agent_profile; ablation in the monitor table (good rate with / without profile on 200 train questions).

## Stage 2 - grading (src/train/react_data.py)
- Good = ROUGE-L >= .6 vs reference and cites a gold Dieu (S1). Others = wrong. Expected usable ~13 % for an untrained model (datamate), so most rows go to repair.

## Stage 3 - DeepSeek repair (src/train/react_fix.py, teacher = deepseekflash instead of Haiku)
- Batches of ~20 repair requests per call (the 8.4k-token Vibe overhead is paid once per batch), pacing a few calls per minute; strict JSON (<= 2 tool calls + final answer) from the first bad step; keep a repair only if re-running its tool call returns the gold Dieu, the answer grades ok and the 13-gram copy share vs the reference <= .5; answer must follow the reference template ("Can cu theo Dieu X <Luat> ... nhu sau:" + verbatim quote).
- Questions still failing after one repair are dropped, not retried more than once.

## Stage 4 - build the one-shot SFT rows (format B)
- Per question: gold reference answer (as now) + verified S1/S2 answers as extra rows (second valid answer). Contexts = what the deployed retriever returns (test-like), with the data tricks: passage order shuffled for ~50 % of rows, ALL gold Dieu injected when missing, absent rate 0 / 15 %, k in [3, 10], shots 0-3, 20 % refusal rows (as v3b). Dev rows tagged split=dev.
- Decontaminate (Decontam); test never in any file.

## Stage 5 - training and evaluation (GPU0/GPU1, one LLM job each)
- Configs, **2 seeds each, no early stop** (user): A = current v3b (existing 3 seeds .6325/.603/.6134), B = v3b + shuffle + inject-all, C = B + 15 % absent, D = C + teacher rows. Train-only models are evaluated on dev300 greedy (paired bootstrap); the final model trains on train+dev and is evaluated once on test.
- Then DPO round(s) from the best SFT as before if D wins.

## Stage monitoring and abort rules (replaces a pilot)
- After the first 100 questions of stages 1-3 print: rollout good rate, repair accepted rate, tokens and cost per question, usable questions. Abort and report if usable < 25 % of questions or projected tokens > 60 M.
- GPU check (nvidia-smi util/temp) every ~10 min while a job runs; fix idle GPUs.

## Expected size of the effect
Ceiling from fixing questions whose gold Dieu is not in the first top-5 is about +.03 ROUGE-L (react notes); data diversity ~ +.01-.02; +.016 per doubling of data. Report as exploratory unless 3-seed means separate by more than the SFT noise (sd .015).

## GPU schedule (maximise use; estimates from the compute log: SFT ~1 h, dev300 greedy eval ~5 min, DPO ~40 min, 9B rollouts ~2.5 h for 2245 q x 2 seeds)
Principle: B and C need no teacher data, so they train while the teacher data is being made. Two jobs at a time (one per GPU); embed :8101 + rerank :8201 stay on GPU1 (~6 GB); RAM/disk guard before every launch; GPU util/temp check every ~10 min.

| Slot | GPU0 | GPU1 | CPU / API |
|---|---|---|---|
| Tonight (GPU0 idle now) | 9B ReAct rollouts train + dev, K=2 (control index; tools via GPU1 servers) | finish full68 (dev, test) | TDD: --shuffle / inject-all / --absent in sft_data; step-0 diagnostics; gold parser fix; licence read |
| D+1 08:00-09:15 | SFT B seed1 -> eval | SFT B seed2 -> eval | Stage 2 grading; Stage 3 DeepSeek repair (paced batches, ~1-2 h) |
| 09:15-10:30 | SFT C seed1 -> eval | SFT C seed2 -> eval | repair continues; Stage 4 build D rows, decontaminate |
| 10:30-12:00 | SFT D seed1 -> eval | SFT D seed2 -> eval | paired bootstrap B/C vs A |
| 12:00-13:00 | DPO round 1 from best SFT | SFT train+dev of the best config (final) | |
| 13:00-14:30 | DPO round 2 + eval (greedy) | final DPO from train+dev SFT | report text, tables |
| 14:30-15:00 | test eval once (greedy, MBR8) | idle -> spare: extra seed for the closest pair | CHANGELOG, compute-log, PDF |

Gaps are filled with the third seed of any pair whose paired difference is within the noise band (.005-.015).

## Track 4 - the ReAct agent as its own flow (user wants an agent, 2026-11-22)
One agent implementation, three uses: (1) data generator (9B + DeepSeek correction, above); (2) a standalone agentic system evaluated on dev300 / test with a big model (DeepSeek-V4 flash or GLM) and with the 9B, reported as an agentic line next to the 2B one-shot (the graded system keeps the < 4B limit); (3) demo / UI option (API already has /v1/chat/completions stream).
Agent loop: profile block -> initial top-5 context -> tools search_law / get_article / kg_related (neutral) -> answer in the reference template -> evidence gate (one nudge) -> verifier on citations (quoted text must appear in a retrieved or fetched Dieu). Backend-agnostic (OpenAI-compatible: local vLLM 9B, or the Vibe/Mistral route for DeepSeek). Dev numbers of the DeepSeek agent also tell the ceiling of the teacher data (answer vs reference ROUGE-L) without a separate pilot.
Prior numbers: raw 9B teacher on dev .47-.49 (not above SFT 2B), trained 2B ReAct .537 / .574; the agent line is expected to need the profile + initial context to compete.

## Student = 2B core agent (user, 2026-11-22) - what changes
- Agent = one-shot plus optional tool calls: the prompt carries the retrieval profile + top-5 context; most questions are answered in one turn (as the one-shot model does), tool calls only when the context lacks the gold. So the agent must not be worse than the one-shot model on questions that need no tool.
- Training rows (multi-turn, loss on assistant turns only, src/train/react_sft.py): (a) one-shot rows in agent format (reference answers, all data tricks: shuffle, inject-all, absent rate, template opener) so direct answering stays strong; (b) tool trajectories S1 (9B good) and S2 (DeepSeek-repaired) for questions where the gold Dieu is not in the initial context; (c) refusal / "context is enough, do not search" rows so the agent does not over-call.
- Configs (2 seeds each): A one-shot v3b (existing), B one-shot + tricks, C B + absent, D one-shot + teacher rows, **E agent SFT = rows (a)+(b)+(c)**, F = E + DPO / multi-turn GRPO if E >= D. Gate: E vs the best one-shot on dev300 (paired bootstrap), split by whether the gold Dieu is in the initial top-5.
- Known risks: trained 2B ReAct was .537 / .574 vs one-shot .633 (dev_clean); datamate: a loop SFT on 1,764 trajectories scored below one-shot; <= 3B students do not always profit from large-teacher data. Mitigation: mixed rows so direct answering is kept, profile + initial context, teacher repairs minimal and verified. Latency (turns) and tool-call rate reported.
