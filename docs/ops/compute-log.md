# Compute log

| Date | Account | GPU | Hours | Cost (USD) | What | Result |
|---|---|---|---|---|---|---|
| 2026-10-31 | ngocle | 1× L4 | ~1.9 | 1.47 | `baseline` (notebook pipeline, HF fp32, test then dev) | ran out of credit at test 36/200; 25 rows saved; FAISS index (176,506 chunks) saved on volume |
| 2026-10-31 | ngocle | 1× L4 | ~1.9 | 1.32 | `embed` c2000 × {bge-m3, vn-emb, e5-large} | bge-m3 + vn-emb done (188,228 chunks, ~33 min each); e5-large cut by credit; c1000 not reached |
| 2026-10-31 | hoangphu | 1× L4 | ~1.1 | 0.88 | `models` sweep on dev | only Qwen1.5-1.8B × baseline prompt × {256, 768} done before credit ran out |
| 2026-10-31 | phu-nguyen | 1× L4 | ~0.4 | 0.33 | `trainprobe` LoRA r64 CPT throughput | done: 1.5B 1,537 tok/s (bs 8), 3B 954 tok/s (bs 4; bs 8 OOM) |
