# Findings — MBR / self-consistency for free text (agent, 2026-11-18; snippets, verify before citing)

- N 4–64; gains grow with N (Jinnai XSum ROUGE-L 30.7 at N=4 → 34.6 at N=64), diminishing ~16–32 (unverified).
- Epsilon sampling (ε≈0.02, T=1) beat top-k / nucleus / ancestral as the MBR pool (Jinnai et al.).
- Utility = the final metric (ROUGE-L for ROUGE-L). Universal Self-Consistency GovReport ROUGE-1: greedy 38.8,
  random 38.5, USC 40.2 (secondary source). Suzgun et al.: +3–7 ROUGE/BLEU.
- Cost: N× generation; O(N²) ROUGE calls, cheap on CPU. Pitfall: length / generic-centroid bias → check length ratio.
- Greedy in the pool: no source; cheap to add (N+1).
Sources: https://arxiv.org/abs/2310.01387 · https://arxiv.org/abs/2311.17311 · https://arxiv.org/abs/2211.07634 ·
https://arxiv.org/abs/2311.05263 · https://arxiv.org/abs/2105.08504
