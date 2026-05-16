# Convenience targets for Dexora development & release tasks.
# Run `make help` for the menu.

.PHONY: help install install-dev fmt lint test test-cov \
        precommit clean sanity release-check all-stages \
        train-stage1 train-scoring posttrain

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

help:                ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?##/ { \
	    printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ----------------------------- environment -----------------------------
install:              ## pip install runtime requirements only.
	$(PIP) install -r requirements.txt

install-dev:          ## pip install + development extras (ruff/black/pytest/...).
	$(PIP) install -e ".[dev]"
	pre-commit install || true

clean:                ## Remove caches and build artefacts.
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info \
	    .coverage coverage.xml htmlcov

# ----------------------------- quality -----------------------------
fmt:                  ## Auto-format with ruff (imports) + black.
	ruff check --fix .
	ruff format .
	black .

lint:                 ## ruff check + black --check (no edits).
	ruff check .
	black --check .

test:                 ## Run the unit tests.
	pytest tests/

test-cov:             ## Run unit tests with coverage report.
	pytest tests/ --cov --cov-report=term-missing

precommit:            ## Run all pre-commit hooks on every file.
	pre-commit run --all-files

sanity:               ## AST-parse every Python file (matches CI sanity step).
	$(PYTHON) -c "import ast, pathlib, sys; \
		bad=[]; \
		skip=('lerobot/','dataprocess/','reference/','ref_collect_hand/', \
		      'ref_collect_mmk/','ref_replay/','_ref_collet/','deploy/', \
		      'data/preprocess_scripts/'); \
		[ast.parse(p.read_text(), filename=p.as_posix()) for p in pathlib.Path('.').rglob('*.py') \
		 if not any(p.as_posix().startswith(s) for s in skip)]; \
		print('OK')"

release-check:        ## Run release-readiness checks (warnings allowed).
	$(PYTHON) tools/release_check.py

release-check-strict: ## Run release-readiness checks (warnings = errors, CI mode).
	$(PYTHON) tools/release_check.py --strict

# ----------------------------- pipeline -----------------------------
all-stages:           ## Run the full 3-stage Dexora pipeline (see run_all_stages.sh).
	bash run_all_stages.sh

train-stage1:         ## Stage-1 only: pretrain on simulation.
	START_STAGE=1 END_STAGE=1 bash run_all_stages.sh

train-scoring:        ## Stage-2 only: Spre -> Shigh -> log-pi -> discriminator.
	START_STAGE=2 END_STAGE=5 bash run_all_stages.sh

posttrain:            ## Stage-3 only: quality-aware post-training.
	START_STAGE=6 END_STAGE=6 bash run_all_stages.sh
