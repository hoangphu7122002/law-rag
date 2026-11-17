PY ?= uv run python
CLI = PYTHONPATH=src $(PY) -m cli
.DEFAULT_GOAL := help

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

data:  ## unzip corpus, clean res.csv, write the fixed split
	$(CLI) data

test:  ## unit tests
	uv run pytest -q tests

lint:  ## ruff
	uv run ruff check src tests

smoke:  ## whole pipeline on 5 questions, BM25 + fake LLM, no GPU
	$(CLI) run configs/exp/S0_smoke.yaml

run:  ## one experiment: make run EXP=configs/exp/B0_baseline.yaml [ARGS="--set retrieve.k=10"]
	$(CLI) run $(EXP) $(ARGS)

score:  ## (re)score a run: make score RUN=data/runs/<dir>
	$(CLI) score $(RUN)

compare:  ## paired bootstrap: make compare A=data/runs/<a> B=data/runs/<b> [METRIC=rouge_l]
	$(CLI) compare $(A) $(B) --metric $(or $(METRIC),rouge_l)

box:  ## remote GPU box: make box ARGS="autosync on|push|setup|serve <model> --gpu 0|run <exp> [--set k=v]|status|stop <s>|pull|tunnel"
	$(CLI) box $(ARGS)

.PHONY: help data test lint smoke run score compare box
