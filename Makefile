SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help install models index lint format typecheck test check demo craft record bench tune secret-scan screenshot clean

help: ## List the commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Create .venv and install the package with dev tools
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e '.[dev]'

models: ## Download the two local models (about 150 MB, free, no account) and refresh the caches
	$(BIN)/ga index --download

index: ## Build the search index (uses local models if downloaded, else the committed cache)
	$(BIN)/ga index --models auto

lint: ## Ruff lint and format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Apply ruff formatting
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: ## mypy in strict mode
	$(BIN)/mypy

test: ## Unit and integration tests with coverage (no network, no key)
	$(BIN)/pytest

check: lint typecheck test ## Everything CI runs

demo: ## Before and After on the 50 questions. Saves docs/demo-output.txt and reports/demo/
	@mkdir -p docs
	$(BIN)/ga demo 2>/dev/null | tee docs/demo-output.txt

craft: ## Rebuild the hand-crafted cassettes from fixtures/handcrafted/answers.yaml
	$(BIN)/ga craft

record: ## Re-record both pipelines from the live API (needs ANTHROPIC_API_KEY, stops at GA_BUDGET_USD)
	$(BIN)/ga eval --pipeline naive --mode record
	$(BIN)/ga eval --pipeline grounded --mode record

bench: ## Time retrieval with the live local models and check the committed caches
	$(BIN)/ga bench

tune: ## Show the thresholds the dev split chooses (add ARGS=--write to save them)
	$(BIN)/ga tune $(ARGS)

secret-scan: ## Regex scan of the full git history for key patterns
	$(BIN)/python scripts/secret_scan.py

screenshot: ## Render reports/demo/report.html to docs/eval-report.png (headless Chrome, throwaway profile)
	@mkdir -p .cache/chrome-profile
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-first-run \
		--user-data-dir="$(CURDIR)/.cache/chrome-profile" --hide-scrollbars --window-size=1400,1180 \
		--screenshot=docs/eval-report.png "file://$(CURDIR)/reports/demo/report.html" 2>/dev/null

clean: ## Remove generated reports, the index and caches (keeps downloaded models)
	rm -rf reports .cache/index.db .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
