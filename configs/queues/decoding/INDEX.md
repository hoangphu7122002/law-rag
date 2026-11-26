# configs/queues/decoding/ (3 queues)

Launch: `PYTHONPATH=src python -m cli box queue configs/queues/decoding/<file> --parallel --gpu N` (see ../README.md). Description = the first comment line of each file.

| file | what |
|---|---|
| `mbr16b_g1.txt` | rerun MBR N=16 on v0: first attempt died with "Too many open files" (16 requests x 16 samples) -> concurrency 4 |
| `mbr2_g0.txt` | MBR variants on dev: N=16 on v0, then v3b (2 shots) MBR8 |
| `mbr_g0.txt` | MBR on SFT v0 (dev, then pre-declared test); after GPU0's GRPO c2 queue ends |
