# Findings: search API alternatives (agent summary from third-party blogs; ALL unverified — confirm on vendor pages before budgeting)
| API | $/1k | free quota | snippets | domain restriction | cache/store terms |
| Brave | ~5 | $5 credit/mo (~1k queries); old 2k-5k free tier ended Feb 2026 | yes | site: in query (unverified) | strictest: no storing/caching beyond transient use unless a storage-rights plan (reportedly enterprise) |
| Tavily | ~8 | 1,000 credits/mo | yes + optional raw content | include_domains (up to 300) | unknown, check ToS |
| Serper | ~1 (down to ~0.30 at volume) | 2,500 one-time | yes (Google SERP) | site: in query | unknown; scraped Google results → gray legal risk |
| Exa | ~4-7 (sources conflict) | $10/mo (one source $20 start) | yes + highlights/text | includeDomains (up to 1,200, vendor claim) | unknown |
| Google CSE JSON | 5 | 100/day | yes | yes | Google terms limit caching; CLOSED to new customers, shuts down 2027-01-01 → not viable |
| DuckDuckGo (unofficial) | free | - | yes | site: | no official API, scraping breaches ToS → not for production |
| SerpAPI | ~15-25 | 250/mo | yes | site: | unknown |
Vietnamese: none of the pages address it; Google-based SERP (Serper, SerpAPI) should be best (gl=vn, hl=vi); Brave/DDG support country=VN, ui_lang=vi (unverified); Tavily/Exa English-centric.
Agent's pick: Tavily with include_domains (free 1k/mo) or Serper with site: OR (cheapest). Brave's terms restrict caching. thuvienphapluat.vn restricts scraping.
Sources: apicostcalc.com/blog/exa-vs-tavily-vs-serper-vs-brave-web-search-api-cost.html · apiserpent.com/blog/free-brave-search-api · codenote.net/en/posts/tavily-alternatives-cost-comparison-search-extract-api/ · exa.ai/versus/tavily · beri.net/article/tavily-vs-exa-vs-brave-vs-perplexity-search-api-agent-cost-2026 · developers.google.com/custom-search/v1/overview · firecrawl.dev/blog/google-search-json-api-alternative
