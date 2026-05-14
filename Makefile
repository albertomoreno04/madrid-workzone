# MADTwin make targets. Every target is one-shot reproducible.

SHELL := /bin/bash
PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n\nTargets:\n"} \
	      /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' \
	      $(MAKEFILE_LIST)

.PHONY: install
install: ## Install the project with dev + eval extras (editable).
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,eval]"

.PHONY: install-all
install-all: ## Install every optional dependency group (heavy).
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[all]"

.PHONY: install-sim
install-sim: ## Install the SUMO Python bindings.
	$(PIP) install -e ".[sim]"

.PHONY: precommit
precommit: ## Install the pre-commit hooks.
	pre-commit install

.PHONY: lint
lint: ## Ruff lint + format check.
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m ruff format --check src scripts tests

.PHONY: format
format: ## Apply Ruff formatting.
	$(PYTHON) -m ruff check --fix src scripts tests
	$(PYTHON) -m ruff format src scripts tests

.PHONY: typecheck
typecheck: ## mypy on the src tree.
	$(PYTHON) -m mypy src

.PHONY: test
test: ## Run the unit test suite.
	$(PYTHON) -m pytest

.PHONY: test-cov
test-cov: ## Run tests with coverage.
	$(PYTHON) -m pytest --cov=madrid_twin --cov-report=term-missing

.PHONY: check
check: lint typecheck test ## Full pre-flight check.

.PHONY: baseline-metrics
baseline-metrics: ## Compute baseline metrics from a SUMO output bundle.
	$(PYTHON) -m scripts.analyze_baseline

.PHONY: probe
probe: ## Hit the Madrid Ayuntamiento traffic intensity feed; save a snapshot.
	$(PYTHON) -m scripts.probe_open_data

.PHONY: fit-baselines
fit-baselines: ## Fit the Phase 2 forecaster baselines on synthetic data and write a report.
	$(PYTHON) -m scripts.fit_baselines

.PHONY: mlflow-ui
mlflow-ui: ## Launch the local MLflow UI on http://localhost:5000.
	$(PYTHON) -m mlflow ui \
		--backend-store-uri sqlite:///mlruns.db \
		--default-artifact-root ./mlartifacts \
		--host 127.0.0.1 \
		--port 5000

.PHONY: clean
clean: ## Remove caches, build artefacts and local MLflow store.
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf build dist *.egg-info src/*.egg-info
	rm -rf mlruns mlartifacts mlruns.db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
