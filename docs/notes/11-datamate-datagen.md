# How datamate generates its SFT data (read-only look at datamate_train/scripts/datagen, 2026-11-19)

- **Unanswerable rows** (`sft_aug_unanswerable.py`): organic 2.3 % → target ~7 % of rows, "without teaching abstain on answerable
  questions". Three sources: organic gold-unanswerable questions; **transplant** (an answerable question moved to a shop that
  records none of the needed concepts, checked three ways: harness facts, keyword scan of tables, no answerable train question of
  that shop needs it); **contrast** rows (the answerable original is kept; unanswerable rows <= 35 % of a concept's rows).
  The teacher (Claude CLI) writes the reason naming what is missing.
- **Shots**: a hand-authored "shot bank" (`shot_bank_*`, `shot_session.py`): an LLM author drives train questions turn by turn
  through the real harness, blind to the gold (PASS/FAIL only), up to 3 fails per question; train data only.
- **Shuffling** (`sft_shuffle*.py`): rulebook sections shuffled with every `§` reference remapped (augmentation of order).
- **Row selection / checks** (`sft_rows*.py`): rows filtered by checks, student rows first, teacher repairs failures.

Applied here (SFT v3, `cli sft-data --shots 0,1,2,3 --k-range 3,10 --oracle .7 --refuse .08`): refusal rate .08 ≈ their 7 %; a refusal
row is ADDED next to the answerable original (the contrast rule); the refusal context is retrieved for an unrelated question of a
different law (the transplant analogue); shots are the nearest decontaminated train Q/A, never the row's own Điều (test can never see
one either). Not applied: authored shot bank, teacher-written refusal reasons (needs an external API: ask first).
