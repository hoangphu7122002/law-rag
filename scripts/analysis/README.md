# scripts/analysis/

diagnostics and reports on finished runs: tail/mid report, cited-group and coverage diagnostics, failure stages, MBR oracle, selectors, gate/expand caches.

Run from the repo root: `PYTHONPATH=src python scripts/analysis/<file>.py ...` (scripts are not a package; their tests add this folder to `sys.path` in `tests/unit/scripts/conftest.py`).

| file | what |
|---|---|
| `appendix_examples.py` | Automatic checks + worked examples of one run, for the report appendix. |
| `cascade_stats.py` | Cascade: MBR only for the least confident share of questions, greedy for the rest. |
| `cite_stats.py` | How much does a wrong cited Dieu cost? Splits rows by whether the prediction cites the same "Điều N" as the reference. |
| `diag_cited_group.py` | Why is ROUGE-L only ~.71 when the model cites the right Điều? Precision/recall of the LCS, length ratio and khoản agreement inside the 'retrieved and cited' group. |
| `diag_coverage.py` | Does the context (retrieved chunks) contain what the reference quotes? Reference 4-gram coverage by (a) the context, (b) the full gold articles (upper bound), |
| `expand_full_article_caches.py` | Retrieval caches whose passages are FULL Điều instead of chunks (no index, no GPU): the quick end-to-end test of 'do not cut long articles'. |
| `gate_expand_caches.py` | Build retrieval-cache files for the gate + widen experiment (no index, no GPU), so the normal eval path runs it with the retrieval step cached. |
| `llm_rerank.py` | Exploratory: can a large LLM (via Mistral Vibe) re-rank the reranker's top candidates better? Dev only. |
| `low_score_analysis.py` | Where do the low-scoring questions fail? Stage of the first failure, per question of a run: |
| `mbr_oracle.py` | Generator ceiling: sample the MBR candidates per dev question, keep them all, score greedy / MBR pick / best-of-n vs the reference. |
| `retr_compare.py` | Paired bootstrap of per-question retrieval hits between two retrieval caches (control vs a chunking variant). |
| `selector_cv.py` | Learned candidate selector, cross-validated on dev: candidates per question = the model answer + "intro + verbatim article" of each of the 5 |
| `tail_cases.py` | CSV of the low-scoring questions of a run, one row per question, for reading by hand. |
| `tailmid_report.py` | Per-run tail/mid report: ROUGE-L by failure stage (low_score_analysis stages, retrieved-not-cited split into 'cites another retrieved Điều' / 'cites none'), |
