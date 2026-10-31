PY ?= uv run python
.DEFAULT_GOAL := help

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

data:  ## unzip corpus, clean res.csv, write the fixed split
	PYTHONPATH=src $(PY) -m cli.data

test:  ## unit tests
	uv run pytest -q tests/unit

lint:  ## ruff
	uv run ruff check src tests

eval:  ## run one experiment: make eval EXP=config/experiments/<x>.yaml
	PYTHONPATH=src $(PY) -m cli.eval $(EXP)

.PHONY: help data test lint eval
