# configs/queues/eval/ (7 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/eval/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `order67_g0.txt` | Order sensitivity of the final model (dpo_a2 greedy, dev300, 2 shots): the same top-5 passages reversed (limit=302) or shuffled (limit=303). Reference: reranker order .6637. |
| `test56_g0.txt` | TEST replication: dpo_a2s2 (second seed of the same recipe) greedy. |
| `test56_g1.txt` | TEST (user 2026-11-20: "chạy test cấu hình này đi"): dpo_a2 greedy (the chosen deployment config), then MBR8 as the accuracy mode. Same retrieval as the dev runs. |
| `test_g0.txt` | TEST (user asked 2026-11-19 "cứ test test xem ntn"): one pre-declared pair, no tuning on it. Reported as test-informed, final-config candidate = v3b (best dev). |
| `test_variants_g0.txt` | TEST-split ROUGE-L of the finished SFT variants (user-requested, 2026-11-25; used to choose the training methods). GPU0 lane; waits for the D smoke build. |
| `testfull_g0.txt` | FULL TEST pass (user 2026-11-19 "đo full trên test"): every trained variant, report only, no selection beyond what dev chose. |
| `testfull_g1.txt` | (no description line) |
