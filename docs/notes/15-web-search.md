# Web search tool (TinyFish) — design, on/off in config (2026-11-19)

Provider: TinyFish Search API (`GET https://api.search.tinyfish.ai`, header `X-API-Key`, key in env `TINYFISH_API_KEY`, results title / snippet / url;
docs.tinyfish.ai/search-api; response shape from docs + third-party guides, NOT yet verified with a live key; rate limits quoted by a blog: 30 search/min).
Built test-first (`src/generate/web.py`, tests in `tests/unit/generate/test_web.py`, `test_react.py`; 98 tests pass). Nothing was sent to the service yet.

Switches (all in the experiment config, off by default):
- `generate.agent_tools`: add `web_search` / `kg_related` to turn a tool on, remove to turn it off.
- `generate.web`: `domains` (allow-list incl. subdomains), `max_results`, `cache` (sqlite, one external call per distinct query, reruns replay it),
  `block_test_answers` (drop any result sharing a 13-gram with a test answer — evaluation integrity; reported in the run).

Measurement arms (each vs. the arm above it, paired bootstrap on dev_clean / dev_strict; nothing is chosen on test):
| arm | agent_tools |
|---|---|
| T0 | search_law, get_article |
| T1 (+KG) | search_law, get_article, kg_related |
| T2 (+web) | search_law, get_article, web_search |
| T3 (+KG +web) | search_law, get_article, kg_related, web_search |
Each with a non-agent control that adds the same number of passages statically (docs/research/react-tools/).

Open points (need the user):
1. Is calling the web at inference allowed by the assignment? Unknown; default off.
2. The questions are sent to an external service when the tool is on (outward-facing); needs the user's go-ahead and a key; no private data is in the query (question text only).
3. Reference answers may come from the web: a result can contain the answer → leak filter above; check which sites the references use before allowing domains.

## Cost decision (docs/research/tinyfish-cost/, 2026-11-19)
- TinyFish official pricing page: **Search $0 per request, Fetch $0 per URL**; limits Search 30/min + 500/hour, Fetch 150 URLs/min + 1,000/day; Agent $0.016/step and
  Browser $0.002/min are the only things that spend the wallet ($12.776) → we use ONLY Search (Fetch later if snippets are not enough). Verify that the wallet balance does not drop.
- Alternatives (Brave ~$5/1k with strict no-cache terms, Tavily ~$8/1k, Serper ~$1/1k, Exa ~$4–7/1k, Google CSE closed to new customers, DuckDuckGo scraping) are not needed.
- Built for the limits (test-first): spacing 60/30 s between paid calls, hourly cap 500 → clear error (cached queries still work), thread-safe cache.
  An experiment of 166 questions × ≤3 searches ≈ 500 calls ≈ 1 hour of quota: run it in chunks; reruns replay the cache.
- Response shape verified with one live call (title, snippet, url, site_name, 10 results). The API key was pasted in the chat: it is NOT stored in the repo; for the box it goes in an env var (decision pending with the user).

## Tier (user's billing page, 2026-11-19)
- Account upgraded to **Builder** ("Pay per use, pay per call"): Search 150/min, Fetch 250 URLs/min, Agent 5 at once (Hobby was 30 / 150 / 2). Hourly / daily caps of Builder not seen yet
  ("View all limits"); `generate.web.calls_per_hour` stays 500 until then. Wallet $12.776 = bonuses (Welcome +$8, Missions +$5) − Agent usage $0.224 (not ours: we only called Search once).
- Unknown: whether Search stays $0 on Builder (pay per call) → watch the wallet after the first runs.

## Correction (billing page "All limits", 2026-11-19) — Search is free only on HOBBY
| | Hobby | Builder |
|---|---|---|
| TinySearch | 30/min, free | 150/min, $0.005 / query |
| TinyFetch | 150/min, free | 250/min, $0.001 / URL |
| TinyAgent | 2 at once, $0.016/step | 5 at once, $0.016/step |
The earlier "Search is free" held for Hobby only; the upgrade to Builder made each query cost money. Wallet $12.776 is bonus credit.
Budget (estimate): one dev_clean arm ≈ 500 queries ≈ $2.5 on Builder; T2+T3 ≈ $5; plus test arms ≈ $3–6 → $10–12.5 for the whole plan, close to the wallet.
Decision: run the experiments on Hobby (free; 500 queries ≈ 1 h at the 30/min + 500/h limits, runs are background anyway); Builder only if speed becomes the bottleneck.
Guard (test-first): `generate.web.cost_per_query` (Hobby 0, Builder 0.005) and `max_cost_usd` — spend is counted per paid call in the cache DB across runs; at the cap the tool returns a clear
"budget reached" error, cached queries still work.

## Runs queued (2026-11-19 18:51), RAW Qwen3.5-2B, dev_clean (166), TinyFish Builder
- `configs/queues/tools_web_g0.txt` (GPU0): T2 (search_law, get_article, web_search) then T3 (+ kg_related); web allow-list vbpl.vn, luatvietnam.vn, thuvienphapluat.vn, vanban.chinhphu.vn;
  150/min, `cost_per_query` 0.005, **`max_cost_usd` 6** (cap over all runs sharing data/cache/web.sqlite), leak filter on.
- `configs/queues/tools_kg_g1.txt` (GPU1): T1 (search_law, get_article, kg_related), no web.
- Both start after queue-1119-180219 (T0 + no-tool control + no-initial-passages), which waits for the four GRPO configs → expected overnight.
- API key: file `/home/app/phunguyen/law_rag_secrets/tinyfish.key` (mode 600, outside the repo; the box account is shared with the datamate team) read via `generate.web.key_file`; never in configs / command lines.

## T0 with / without the SFT adapter (user request, 18:54)
- Raw: queue-1119-180219 (no-tool control, ReAct T0 with 5 initial passages, ReAct T0 with no initial passages) on `Qwen/Qwen3.5-2B`.
- With adapter: queue-1119-185450 (GPU1) first runs the SAME T0 on `q35_v0` (SFT LoRA, merged; no separate adapter file), then T1 (+KG) on the raw model. The T0 non-agent control for v0 is already A4_q35_off (dev_clean .5828, dev_strict .5569).
- Question answered: does SFT keep or break native tool calling (the SFT rows contain no tool calls), and how much of the raw model's gap does the SFT style close? Metrics: rougeL on dev_clean / dev_strict, tool-call count, empty answers.
