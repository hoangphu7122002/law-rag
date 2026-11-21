# Teacher-data plan: 9B ReAct + DeepSeek correction -> 2B one-shot SFT (decided 2026-11-22, run from 2026-11-23, no pilot)

Decisions (user + defaults): teacher = DeepSeek-V4 flash via /vibe (alias deepseekflash); rollouts by base Qwen3.5-9B (HF hub, same family as the 2B; not the codeact checkpoints); scope = ALL train (2245) + dev (300), never test; TWO approaches (user, 2026-11-22), both with a < 4B 2B student, built and reported side by side: Approach 1 = one-shot 2B; Approach 2 = ReAct 2B core agent. Neither is only a control for the other; web_search IS a tool of the agent (user, 2026-11-22: the closed store lacks some information, e.g. TB 6150, the 2025 draft of TT 39/2016, the TT 10/2025 appendix), with the guards of "Web tool" below.

## Stage 0 - preconditions (morning)
- Index full68 (corpus + web_new_laws) finished and compared with control 4ac2a75c (retr_compare); use the better one for tools and training context.
- Box guard: RAM >= 14 GB, disk >= 20 GB free, GPU temp < 85 C; embed :8101 + rerank :8201 on GPU1 up; one heavy job at a time.
- Licence (partial, record in the report): DeepSeek V4 weights reported MIT (Alibaba Cloud listing, secondary news); DeepSeek Open Platform ToS allows using outputs to train other models incl. distillation; V4 LICENSE file not read; the call goes through Mistral Vibe, whose terms were not read either. Read both before the full run; disclose in the report.
- Budget cap: total --max-price 40 USD for the teacher; ledger project label `tdata`; price of deepseekflash is not listed (ledger shows 0), so cost is tracked in tokens (stop at 60 M tokens total).

## Stage 1 - 9B ReAct rollouts (GPU0)
- vLLM Qwen3.5-9B bf16 on GPU0 (~18 GB), tools search_law / get_article / kg_related over the chosen index (embed/rerank on GPU1), max_turns 4, K = 2 rollouts per question at T 0.7, non-thinking.
- Output data/gen/tdata/rollouts_{train,dev}.jsonl, one line per rollout with trajectory, tool results, final answer.

## Stage 1a - evidence from the old ReAct data and the resulting rollout rule (2026-11-22)
Old trajectories (train 2245, initial top-5): raw 9B with 3 tools called a tool in 100 % of the questions (1.47 calls on average) although the gold Dieu was already in the initial prompt for 89 %; good rate 21-22 % (ROUGE >= .6 and cites gold), mean ROUGE .44; by calls: 1 -> .46, 2 -> .47, 3 -> .30 (14 % hit the turn cap). The 2B q35_v0 never called a tool (0.00) and scored .60 / 53-54 % good.
Rule: tools are not offered on the first pass. Pass 1 = 9B answers from profile + top-5 without tools (one-shot style, template opener). Graded. Only the questions that fail (gold not in context, or ROUGE < .6) go to the tool rollout (rounds of Stage 1b) with the web tool allowed. This keeps the tool trajectories focused on real context gaps and gives the 2B agent its "answer directly when the context is enough" rows (c) for free.

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

## Stage 1c-2 - question profile: regex first, LLM fallback (user, 2026-11-22; datamate-style preprocess)
- Layer 1, regex / rules (scalable, general): a registry of extractors, each returns (field, value, confidence); law numbers in all formats (ND-CP, TT-NHNN, QD, QH15, BTC ...), laws named by title + year resolved through the title index (as the gold parser does), Dieu / khoan / diem and ranges, years, question type from a vocabulary, flags (du thao, mau, phu luc, thong bao, hien hanh). Patterns are data (tables + law_key), parametrised tests on real train questions, coverage report script (share of questions with each field).
- Completeness check: if law and Dieu and type are all missing, or the question is long / compound, go to layer 2.
- Layer 2, LLM fallback: one short extraction call returning strict JSON {laws:[{number|name}], dieu:[], type, flags, key_terms}. Validation: schema, type from the fixed vocabulary, a law number must match law_key and exist in the store or KG else it is dropped (no invented numbers), result marked unverified; cached by question hash. Model: a SMALL one (user): the 2B (Qwen3.5-2B base or the student) both in the teacher rollouts and at agent inference, JSON-constrained decoding (vLLM structured outputs, to verify in the installed version); never the 9B, never an external API. If the raw 2B extracts badly, distil it from DeepSeek labels on train questions (cheap, questions only). The LLM sees the question only, never a reference.
- Measure: fallback rate; law extraction precision (a law named in the question vs the gold law of the reference); share of questions with any specific field; profile on / off ablation in the agent. Question rewriting stays out of the scored one-shot system; key_terms may feed search_law in the agent (ablated).

## Stage 2 - grading (src/train/react_data.py)
- Good = ROUGE-L >= .6 vs reference and cites a gold Dieu (S1). Others = wrong. Expected usable ~13 % for an untrained model (datamate), so most rows go to repair.

## Stage 3 - DeepSeek repair (src/train/react_fix.py, teacher = deepseekflash instead of Haiku)
- Batches of ~20 repair requests per call (the 8.4k-token Vibe overhead is paid once per batch), pacing a few calls per minute; strict JSON (<= 2 tool calls + final answer) from the first bad step; keep a repair only if re-running its tool call returns the gold Dieu, the answer grades ok and the 13-gram copy share vs the reference <= .5; answer must follow the reference template ("Can cu theo Dieu X <Luat> ... nhu sau:" + verbatim quote).
- Questions still failing after one repair are dropped, not retried more than once.

## Stage 3b - synthetic generation besides repair (user: also generate new, diverse, hard / noisy samples)
Source: corpus articles (law + Dieu, whole families), excluding every gold article of dev/test; DeepSeek writes question + template answer (verbatim quote, "Can cu theo Dieu X ... nhu sau:"), then the same verification (quote must appear verbatim in the article, ROUGE of the quote vs the article, copy guard).
Patterns (target mix of the synthetic share, about 30-40 % of all rows, never above 50 %): single article 40 %; multi-article same law 20 %; multi-law 10 %; version family (original + amending text, question depends on which applies) 10 %; notice / appendix / form / draft (web-sourced original documents) 5 %; vague or colloquial paraphrase of a question 10 %; no-answer-in-context 5-10 % (refusal / tool call).
Hard and noisy contexts are built from the retriever's real mistakes, not random noise: sibling Dieu of the same law, other version of the same family ranked first, similar-topic law, shuffled order, random k in [3, 10], gold dropped from the context. For the agent a dropped-gold context gives an exact label "context lacks gold" -> trajectory search_law / get_article -> answer (the recovery case, with a known correct target); contexts that keep the gold give the "answer directly" rows.
Controls: question-style check (embedding / n-gram distance to real train questions; drop the far tail), near-duplicate filter against dev/test questions and answers (13-gram + embedding), style-fit check on articles that also have a real reference, cap per article and per law so a few big laws do not dominate; noise strength tuned by the same 2-seed comparison (v3b already uses 20 % refusal rows).

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

## GPU schedule (maximise use; estimates from the compute log: SFT ~1 h, dev300 greedy eval ~5 min, DPO ~40 min, 9B rollouts ~2.5 h for 2245 q x 2 seeds; agent eval is multi-turn, slower)
Two lanes: GPU0 = Approach 1 (one-shot), GPU1 = Approach 2 (agent) once its data exists. B and C need no teacher data, so they train while the teacher data is made. One job per GPU; embed :8101 + rerank :8201 stay on GPU1 (~6 GB); RAM/disk guard before every launch; GPU util/temp check every ~10 min.

| Slot | GPU0 (one-shot) | GPU1 (agent / spare) | CPU / API |
|---|---|---|---|
| Tonight (GPU0 idle now) | 9B ReAct rollouts train + dev (rounds 1-2) | finish full68 (dev, test) | TDD: build_profile, trajectory monitor, shuffle / inject-all / absent; step-0 diagnostics; gold parser fix; licence read |
| D+1 08:00-09:15 | SFT B s1 -> eval | SFT B s2 -> eval | grading; DeepSeek repair (paced) |
| 09:15-10:30 | SFT C s1 -> eval | SFT C s2 -> eval | repair continues; build D rows and agent rows; decontaminate |
| 10:30-12:00 | SFT D s1 -> eval | SFT E s1 (agent) -> eval | bootstrap B/C vs A |
| 12:00-13:30 | SFT D s2 -> eval | SFT E s2 (agent) -> eval | |
| 13:30-15:00 | DPO from best one-shot | agent F (DPO / GRPO) if E improves, else spare seed | |
| 15:00-16:30 | final one-shot SFT on train+dev, then DPO | final agent on train+dev | report text, tables |
| 16:30-17:30 | test once: one-shot greedy + MBR8 | test once: agent | CHANGELOG, compute-log, PDF |

Gaps are filled with the third seed of any pair whose paired difference is within the noise band (.005-.015).

## Track 4 - the ReAct agent as its own flow (user wants an agent, 2026-11-22)
One agent implementation, three uses: (1) data generator (9B + DeepSeek correction, above); (2) a standalone agentic system evaluated on dev300 / test with a big model (DeepSeek-V4 flash or GLM) and with the 9B, reported as an agentic line next to the 2B one-shot (the graded system keeps the < 4B limit); (3) demo / UI option (API already has /v1/chat/completions stream).
Agent loop: profile block -> initial top-5 context -> tools search_law / get_article / kg_related (neutral) -> answer in the reference template -> evidence gate (one nudge) -> verifier on citations (quoted text must appear in a retrieved or fetched Dieu). Backend-agnostic (OpenAI-compatible: local vLLM 9B, or the Vibe/Mistral route for DeepSeek). Dev numbers of the DeepSeek agent also tell the ceiling of the teacher data (answer vs reference ROUGE-L) without a separate pilot.
Prior numbers: raw 9B teacher on dev .47-.49 (not above SFT 2B), trained 2B ReAct .537 / .574; the agent line is expected to need the profile + initial context to compete.

## Approach 2 - the 2B core agent (what it is and how it is trained)
- Agent = one-shot plus optional tool calls: the prompt carries the retrieval profile + top-5 context; most questions are answered in one turn (as the one-shot model does), tool calls only when the context lacks the gold. So the agent must not be worse than the one-shot model on questions that need no tool.
- Training rows (multi-turn, loss on assistant turns only, src/train/react_sft.py): (a) one-shot rows in agent format (reference answers, all data tricks: shuffle, inject-all, absent rate, template opener) so direct answering stays strong; (b) tool trajectories S1 (9B good) and S2 (DeepSeek-repaired) for questions where the gold Dieu is not in the initial context; (c) refusal / "context is enough, do not search" rows so the agent does not over-call.
- Configs (2 seeds each): A one-shot v3b (existing), B one-shot + tricks, C B + absent, D one-shot + teacher rows, **E agent SFT = rows (a)+(b)+(c)**, F = E + DPO / multi-turn GRPO if E improves. Compared with the best one-shot on dev300 (paired bootstrap), split by whether the gold Dieu is in the initial top-5; both approaches are reported whatever the outcome.
- Known risks: trained 2B ReAct was .537 / .574 vs one-shot .633 (dev_clean); datamate: a loop SFT on 1,764 trajectories scored below one-shot; <= 3B students do not always profit from large-teacher data. Mitigation: mixed rows so direct answering is kept, profile + initial context, teacher repairs minimal and verified. Latency (turns) and tool-call rate reported.

## Web tool (user decision 2026-11-22: use web search when the closed store lacks the information)
- Infrastructure exists (src/generate/web.py, docs/notes/15-web-search.md): TinyFish Search ($0 per request, 30 / min, 500 / hour; Fetch $0 per URL, 1,000 / day), sqlite query cache, domain allow-list, leak filter (13-grams of blocked texts). Key only via env TINYFISH_API_KEY (the pasted key must be rotated by the user).
- Tools of the agent: search_law, get_article, kg_related, web_search (+ fetch for an allowed document page when the snippet is not enough). The prompt says: use the store first; web only when the store has no such document / appendix / notice / draft.
- Guards (code, TDD): allow-list of hosts for original texts (vbpl.vn, chinhphu.vn incl. datafiles, sbv.gov.vn, and document pages of thuvienphapluat / luatvietnam) plus a path deny-list for advice / news / Q&A pages (hoi-dap, tin-tuc, bai-viet-lien-quan ...), because the references are Q&A-style articles from the same sites; blocked_texts = the reference answers of train, dev and test (13-gram leak filter; it does not catch paraphrases, so the path deny-list stays); results dated after 2025 are flagged (web returns the newest consolidated versions, the references were written in 2025). Every web call and URL is logged in the trajectory.
- Rate budget: 500 searches / hour is a schedule constraint. Web calls only when the model asks (and in rounds 2-3), cache on, expected ~1,000 calls overall = 2-3 h of rate limit; run web rollouts as their own queue so the other lanes do not wait.
- Reporting: two lines, closed store and web-enabled (arms T0..T3 of note 15); no web at test time for the one-shot approach. Whether calling the web at inference is allowed by the assignment is still unconfirmed; the web-enabled agent is reported as an extra line, never replacing the closed-store result.
- Training: trajectories that used web (S1/S2) are kept with their tool outputs; the student learns when to call it; the closed-store agent is also trained without web rows to compare.
