# 01 — Research: directions to beat the baseline (2026-10-31)

Web research (2 rounds, ~17 pages; evidence trail not kept, only this summary). Our own data facts first,
because they change the ranking.

## Our data facts (measured, `make data` split seed 20261031)

- Of 142 test questions with a law id, **93 %** cite a law that also appears in `train`, **57 %** the same
  (law, Điều) pair. Most cited laws (2024–25 banking circulars) are **not in the corpus**.
- 54 % of reference answers start with "Căn cứ"; references are long (median 1,426 chars).

## Ranked directions

| # | Direction | Expected gain | Cost | Risk | Evidence |
|---|---|---|---|---|---|
| 1 | LoRA SFT (RAFT style: question + gold/neighbour context + distractors → template answer) on `train` | large (style + length + law knowledge) | ~1–3 USD Modal, < 1 GPU-h on L4/A10G | low | VLQA: LLM answers shorter than gold; CRAFT = RAFT + LoRA |
| 2 | Train-QA memory: retrieve nearest train pairs, put their answers in context | medium–large (only source for laws missing from corpus) | ~0 | leakage (exclude self), copying wrong article | our 93 % / 57 % overlap; no paper found |
| 3 | Hybrid BM25 (pyvi segmentation) + dense, chunks by Điều/Khoản, reranker | medium | low | corpus ceiling | arXiv 2409.13699 hybrid R@1 0.78 vs 0.65/0.74; VLQA reranker MAP@2 0.492→0.550 |
| 4 | Better embedder (AITeamVN Vietnamese_Embedding 0.6B, bge-m3) + Vietnamese_Reranker | small | low | self-reported numbers | AITeamVN cards (Zalo legal Acc@1 0.727 vs bge-m3 0.568) |
| 5 | Graph RAG: cheap regex citation/structure graph (law → Điều → khoản, "căn cứ", "sửa đổi") linking corpus chunks and train QA | small–medium, distinctive for the report | low (no LLM extraction) | untested | CEUR Vol-4079: LightRAG 0.86 vs naive 0.82, HippoRAG2 0.78, LLM-built graphs slow and need strong LLMs |
| 6 | Legal-domain small models (ViLegalQwen3-1.7B) | unknown | low | no scores | ViLegalLM repo |

## Generator candidates (< 4B)

Qwen2.5-3B-Instruct 3.09B (safe default) · Qwen3-1.7B 1.7B · Vi-Qwen2-3B-RAG ~3.1B · Qwen3-4B is 4.0B → **not** < 4B.

## Sources

VLQA https://arxiv.org/html/2507.19995v1 · CRAFT https://arxiv.org/abs/2409.17648 ·
hybrid report https://arxiv.org/pdf/2409.13699 · VLSP 2025 DRiLL https://aclanthology.org/2025.vlsp-1.19.pdf ·
VLSP 2025 SLM https://aclanthology.org/2025.vlsp-1.23.pdf · Graph RAG legal benchmark https://ceur-ws.org/Vol-4079/paper6.pdf ·
AITeamVN https://huggingface.co/AITeamVN/Vietnamese_Reranker · ViLegalLM https://github.com/ntphuc149/ViLegalLM
