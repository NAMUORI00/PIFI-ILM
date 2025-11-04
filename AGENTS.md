# Repository Guidelines

## Project Structure & Module Organization
- `Classification/` — classification tasks; entrypoint `Classification/main.py`.
- `TextualEntailment/` — entailment tasks; entrypoint `TextualEntailment/main.py`.
- `requirements.txt` — pinned runtime deps (incl. PyTorch + CUDA 11.8 wheels).
- Data/model paths default to NAS-like locations; pass CLI flags to use local folders (e.g., `--data_path ./data --model_path ./models`).

## Virtual Environment & Setup
- Create and activate a venv:
  - `make venv` then `source .venv/bin/activate`
- Install dependencies (CUDA 11.8 wheels by default):
  - `make install`
- CPU-only install (no CUDA):
  - `make install-cpu`
- Optional dev tools (formatter/linter/tests):
  - `make dev-tools`

## Build, Test, and Development Commands
- Smoke test environment: `make test`
- Run classification: `make classify ARGS="--job=training --task_dataset=sst2 --model_type=bert --method=pifi"`
- Run entailment: `make entail ARGS="--job=training --task_dataset=mnli --model_type=bert --method=base"`
- Clean artifacts/venv: `make clean`

## Coding Style & Naming Conventions
- Python 3.10+ recommended. Use 4‑space indentation and type hints when feasible.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Formatting: prefer `black` (via `make dev-tools`) and lint with `ruff`. Keep functions small and focused; avoid hard‑coded absolute paths.

## Testing Guidelines
- Framework: `pytest` (install via `make dev-tools`).
- Place tests under `tests/` with files named `test_*.py`.
- Example: `pytest -q` after activating venv. Add quick data‑free unit tests for utilities; use small fixtures/mocks for task code.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (≤72 chars), meaningful body when needed. Group related changes; avoid noisy reformat‑only commits.
- PRs: include purpose, key changes, run commands, and any dataset/path assumptions. Link issues and add logs/metrics tables or screenshots when relevant.

## Security & Configuration Tips
- Override default NAS paths via CLI flags (`--data_path`, `--model_path`, `--result_path`, etc.).
- Do not commit datasets, model weights, credentials, or WANDB tokens. Add large local paths to `.gitignore` as needed.
