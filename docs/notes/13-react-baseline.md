# ReAct baseline on the raw Qwen3.5-2B (2026-11-19)

Source: datamate `blog-draft/posts/datamate/05-react-then-codeact.md` (text-to-SQL on tables, Qwen3.5-9B). What carries over:
- Run the loop on the BASE model with Qwen3.5's native `<tool_call>` format (no fine-tune): their base loop 0.579 → 0.635 with loop fixes; a fine-tuned
  loop on 1,764 trajectories (ca1) scored 0.346, below one-shot 0.476. Train on trajectories only after the base baseline exists.
- Loop guards (their ca1: 42 % of turns errored, 51/96 empty answers from repeats, 37 from context overflow): skip repeated calls, remove tools on the last
  turn + "answer now", cut tool results, stable seed per question, tool errors returned as text. All in `src/generate/react.py`, 5 unit tests.
- Noise: same config twice differed 0.635 / 0.622 (13 % of answers changed) → judge by paired bootstrap on dev_clean/dev_strict, never one run.
- Vote with abstain was their big gain (+.117); abstaining/refusing hurts our lexical metrics (refusal rougeL .066) → our analogue is MBR (+.028 test).
Not portable: tools = pandas/SQL views; ours = search_law / get_article over the legal corpus; their metric rewards abstain.

Design: `generate.agent: true` → `ReactGenerator` (first prompt = normal vi_rag prompt with `agent_initial_k` cached passages + tool instructions);
tools `search_law(query)` (live retrieve+rerank) and `get_article(law, dieu)` (lookup by (law key, Điều) in the index docstore).
vLLM: `serve <model> --gpu N --tools` adds `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
Arms (configs/exp/A7_react.yaml, dev_clean 166): raw model no tools; ReAct + 5 initial passages; ReAct without initial passages.
Later arms: + KG tools (rule-built cites/amends graph), + web search (only if the assignment allows it at inference; cache + leak filter), then SFT on trajectories.

## Tool-design research (2026-11-19, research_react_tools/, weak evidence, snippets only)
- No clean accuracy-vs-tool-count study for 1–4B; a vendor report has an off-the-shelf Qwen3.5-4B at ~6 % with dozens of tools, 71 % after fine-tuning. Small models (3B–14B) gain from tools mainly after RL/SFT on tool use (Search-R1 +20 % at 3B), prompting alone is weak.
- Agents over-search (20–28 % of search steps unnecessary) and under-search; iteration helps multi-hop, adds noise on simple lookups; no single-shot vs iterative result on legal QA, none for Vietnamese or <4B.
- Graph/neighbour expansion gains are confounded by retrieval depth (use a quantity-matched control, k=10); no quantified gain for amendment tools or web search.
- Decision: keep the tool set minimal; test tools ONE AT A TIME, cap turns at 2–3, and compare each tool against a non-agentic control (same extra passages added statically).

## Optional answer gate (`generate.check`, 2026-11-19; empty = off)
- datamate has three answer layers (vote → model checker with quoted evidence + repair guards → code rules that turn grey answers into ABSTAIN, each rule calibrated on train gold:
  a rule firing on > 5 % of correct answers stays off). Ours is a GATE (pass / fail) whose action is never "refuse" (refusal rougeL .066): in MBR failing candidates are dropped before
  the consensus pick (all fail → keep all); in ReAct a failing final answer gets ONE nudge, the next answer is final.
- First check `grounded_citation`: every "Điều N" in the answer must occur in the context (passages + tool results). v0 greedy dev: 21/300 answers (7 %) fail it, rougeL .303 vs .625 for the rest.
- Calibration caveat: the REFERENCE answers fail the same gate vs the retrieved top-5 in 51/300 dev (17 %) and 33/200 test (16.5 %) — mostly questions whose gold Điều was not retrieved — so by datamate's 5 % rule
  it must not be a veto; it is used only as a preference (MBR filter with fallback, one nudge). Whether it helps is an experiment, not an assumption.
- Tests: tests/unit/generate/test_check.py, test_generator.py (MBR), test_react.py (nudge); 118 pass.
