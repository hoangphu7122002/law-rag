# kNN prior from train questions (2026-11-19/20) — candidate recall only, no generation yet

Idea: for a question, embed it (AITeamVN), take the m nearest TRAIN questions (2,245 train rows with gold) and add their gold (law, Điều) pairs to the candidates.
Metric: share of questions whose gold pair is among the pairs (retrieved top-5) / (prior pairs) / (union). Retrieval = C1's dense + rerank top-5.

| split | n | retr@5 | prior m=1 / 5 / 10 | union m=1 / 3 / 5 / 10 | avg extra pairs (m=5) |
|---|---|---|---|---|---|
| dev (all) | 264 | .905 | .436 / .614 / .663 | .932 / .943 / .955 / .966 | 6.0 |
| dev_clean | 130 | .908 | .246 / .369 / .400 | .923 / .938 / .938 / .946 | 6.1 |
| dev_strict (no Điều shared with train) | 61 | .918 | .279 / .393 / .393 | .934 / .951 / .951 / .951 | 6.0 |
| test (all; user asked 2026-11-20, a further test look) | 177 | .898 | .520 / .661 / .706 (m=1/5/10) | .938 / .949 / .949 / .955 | 5.7 |

Facts behind it: 2,745 QA rows cover only 365 laws; 95 % of dev questions' primary law occurs in train; 76 % of dev (77 % of test) questions share >= 1 gold Điều with a train question.
Reading: recall rises by 1.5-3.3 points on the clean/strict dev subsets, 4-6 points on the full dev/test (inflated by Điều shared with train, which the clean/strict subsets exclude).
Not measured: effect on rougeL (needs the prior inside the prompt/rerank; recall is not the bottleneck — the model cites the wrong Điều even when the gold is in the top 5).
Caveat: exploits the course dataset's repetition of Điều; if the grader's questions use unseen Điều it does nothing. Disclose as such. Test looked at once for this idea (candidate recall only).
