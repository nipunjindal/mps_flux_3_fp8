# mps_flux_3_fp8

FLUX 3 Image Klein with diffusers 0.43 on Apple Silicon (MPS): bf16 on torch 2.14, FP8 on torch 2.15. Built on a uv/ruff/ty/pre-commit template.

## Quick Reference

```bash
make install        # Create .venv-torch215 + install deps (including dev)
make kernels        # Build .venv-torch214 + .venv-torch215 and register Jupyter kernels
make bench          # Synthetic FLUX 3 transformer step benchmark, bf16 vs FP8, both torch versions
make test           # Run pytest
make lint           # Format + lint + type-check (ruff, shfmt, toml-sort, ty)
make type-check     # ty check only
make pre-commit-run # Run all pre-commit hooks
make clean          # Remove both venvs and caches
make rename NEW=x   # Rename package from python_base to x
```

## Project Layout

- `src/mps_flux_3_fp8/` — Main package: `flux3.py` (pipeline load/generate/metrics), `fp8.py` (fp8r quantization + MPS matmul backends)
- `notebooks/` — `01_flux3_bf16_mps.ipynb` (torch 2.14 kernel), `02_flux3_fp8_mps.ipynb` (torch 2.15 kernel); outputs go to `outputs/` (gitignored)
- `tests/` — Pytest test files
- `scripts/` — Standalone scripts (`bench_transformer.py`, `sweep_fp8_gemm.py`)

## Tooling

- **Package manager**: uv (not pip/poetry)
- **Linter/formatter**: ruff (line-length 120, target py311)
- **Runtime**: diffusers 0.43.0, transformers >= 5.23, torch 2.14.1 / 2.15.0 as mutually exclusive extras (`torch214`, `torch215`)
- **Type checker**: ty
- **Tests**: pytest (testpaths=tests, pythonpath=src)
- **Pre-commit**: ruff-format, ruff lint, shfmt, toml-sort

## Agent Team

Agent definitions live in `.claude/agents/` (orchestrator, implementer, reviewer). When asked to "pick up issue(s)" or "start polling for issues", follow the orchestrator workflow directly in the main conversation — do not spawn it as a sub-agent.

## Conventions

- Run `make lint` before committing — pre-commit hooks enforce this
- All source goes under `src/mps_flux_3_fp8/`; tests mirror the source structure in `tests/`
- Python >=3.11 required
- Keep this file updated as the project evolves — new deps, conventions, or structural changes should be reflected here

## Coding Standards

- Functional core, imperative shell — FP patterns internally, straightforward public APIs
- Code duplication is often better than the wrong abstraction
- Errors-as-types / errors-as-return-values over try-except; nested try-except is prohibited
- Composition over inheritance
- Early returns and sensible defaults over if-else
- Prefer match-case over if-elif-else
- Liberal use of asserts to affirm expected state
- Global import statements only
- Modern Python 3.10+ type annotations (`list[int]` not `List[int]`)
- Prefer Literals over string enumerations
- Products are `@dataclass` (preferably frozen), sums are `|`, eliminate with `match`
- Protocols and ABCs for decoupling and type lambdas
- Docstrings with argument types, return types, and exceptions; include tensor shapes if applicable
- Liberal logging for observability; never use f-strings in log calls — use structured logging or logger formatting
- Be judicious with comments — skip the obvious, preserve the complex
