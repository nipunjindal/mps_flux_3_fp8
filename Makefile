.PHONY: install install-214 install-215 kernels bench test lint type-check clean rename pre-commit-run

# Two virtual environments: torch 2.14 (bf16 baseline) and torch 2.15 (MPS FP8). Tooling defaults to 2.15.
VENV ?= .venv-torch215
VENV_214 := .venv-torch214
VENV_215 := .venv-torch215

# Default rule
all: install

# Create the virtual environment and install project (including dev deps)
install: $(VENV)

$(VENV): pyproject.toml
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync --extra torch215 --extra dev
	@if [ -d .git ]; then $(VENV)/bin/pre-commit install; fi
	touch $(VENV)

install-214:
	UV_PROJECT_ENVIRONMENT=$(VENV_214) uv sync --extra torch214 --extra dev

install-215:
	UV_PROJECT_ENVIRONMENT=$(VENV_215) uv sync --extra torch215 --extra dev

# Register both venvs as Jupyter kernels used by notebooks/01 (torch 2.14) and notebooks/02 (torch 2.15)
kernels: install-214 install-215
	$(VENV_214)/bin/python -m ipykernel install --user --name flux3-torch214 --display-name "FLUX3 (torch 2.14)"
	$(VENV_215)/bin/python -m ipykernel install --user --name flux3-torch215 --display-name "FLUX3 (torch 2.15)"

# Synthetic one-step transformer benchmark (random weights, no checkpoint needed) on both torch versions
bench: install-214 install-215
	$(VENV_214)/bin/python scripts/bench_transformer.py
	$(VENV_215)/bin/python scripts/bench_transformer.py

# Run the test suite using pytest
# (pytest must be in the dev dependencies of pyproject.toml)
test: install
	$(VENV)/bin/pytest -q

# Lint: format code, run lint checks, and perform type checking in one go
lint: install
	$(VENV)/bin/ruff format .
	find . -path './.venv' -prune -o -name '*.sh' -exec $(VENV)/bin/shfmt -i 4 -w {} +
	$(VENV)/bin/toml-sort -i pyproject.toml
	$(VENV)/bin/ruff check --fix --select I .
	$(VENV)/bin/ruff check --fix
	$(VENV)/bin/ty check .

# Stand-alone type checking target (optional)
type-check: install
	$(VENV)/bin/ty check .

# Run pre-commit on all files (useful for CI or manual checks)
pre-commit-run: install
	$(VENV)/bin/pre-commit run --all-files

# Remove the virtual environment and other caches
clean:
	rm -rf $(VENV_214) $(VENV_215) **/__pycache__

# ------------------------------------------------------------------------------
# Project scaffolding helpers
# ------------------------------------------------------------------------------

# Usage: make rename NEW=my_cool_lib
# Replaces occurrences of the default package name (python_base) in pyproject and
# source tree and moves the package directory. Run once immediately after
# cloning the template.
rename:
	@if [ -z "$(NEW)" ]; then \
		echo "ERROR: provide a new package name, e.g. 'make rename NEW=my_lib'"; \
		exit 1; \
	fi
	@echo "Renaming python_base -> $(NEW)"
	@grep -RIl 'python_base' pyproject.toml src tests | xargs sed -i "s/python_base/$(NEW)/g"
	@mv src/python_base src/$(NEW)
	@echo "Rename complete. Remember to run 'make clean install' to rebuild the venv."
