# Teacher-data plan: 9B ReAct + DeepSeek correction -> 2B one-shot SFT (decided 2026-11-22, run from 2026-11-23, no pilot)

Decisions (user + defaults): teacher = DeepSeek-V4 flash via /vibe (alias deepseekflash); rollouts by base Qwen3.5-9B (HF hub, same family as the 2B; not the codeact checkpoints); scope = ALL train (2245) + dev (300), never test; student format B = one-shot 2B (deployed system unchanged, < 4B); no web tool (leak risk).

## Stage 0 - preconditions (morning)
- Index full68 (corpus + web_new_laws) finished and compared with control 4ac2a75c (retr_compare); use the better one for tools and training context.
- Box guard: RAM >= 14 GB, disk >= 20 GB free, GPU temp < 85 C; embed :8101 + rerank :8201 on GPU1 up; one heavy job at a time.
- Licence (partial, record in the report): DeepSeek V4 weights reported MIT (Alibaba Cloud listing, secondary news); DeepSeek Open Platform ToS allows using outputs to train other models incl. distillation; V4 LICENSE file not read; the call goes through Mistral Vibe, whose terms were not read either. Read both before the full run; disclose in the report.
- Budget cap: total --max-price 40 USD for the teacher; ledger project label `tdata`; price of deepseekflash is not listed (ledger shows 0), so cost is tracked in tokens (stop at 60 M tokens total).

## Stage 1 - 9B ReAct rollouts (GPU0)
- vLLM Qwen3.5-9B bf16 on GPU0 (~18 GB), tools search_law / get_article / kg_related over the chosen index (embed/rerank on GPU1), max_turns 4, K = 2 rollouts per question at T 0.7, non-thinking.
- Output data/gen/tdata/rollouts_{train,dev}.jsonl, one line per rollout with trajectory, tool results, final answer.

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
