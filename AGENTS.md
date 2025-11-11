# Repository Guidelines

## Project Structure & Module Organization
- `main.py` — unified entry point (recommended).
- `core/` — shared arguments (`core/arguments.py`), utilities (`core/utils.py`), pipeline (`core/pipeline.py`).
- `tasks/` — task proxies to legacy modules (classification → `Classification/`, entailment → `TextualEntailment/`).
- `Classification/`, `TextualEntailment/` — legacy implementations (kept functional).
- `selection/ilm_direct.py` — ILM auto layer selection (PC patching) with reproducible flags.
- `scripts/` — runners:
  - Unified: `scripts/run_experiments.sh` (root-path layout; recommended)
  - Legacy-style (root-path layout):
    - Classification: `run_classification_base.sh`, `run_classification_pifi_last.sh`, `run_classification_pifi_ilm.sh`
    - Entailment: `run_entailment_base.sh`, `run_entailment_pifi_last.sh`, `run_entailment_pifi_ilm.sh`
  - Legacy (kept): `run_classification_ilm.sh`, `run_entailment_ilm.sh`, `run_classification_last.sh`, `run_entailment_last.sh`
- `requirements.txt` — CUDA 11.8 PyTorch wheels (+ ecosystem) pinned.
- Root-path layout for artifacts by default: `cache/`, `preprocessed/`, `models/`, `checkpoints/`, `results/`, `tensorboard_logs/`.

## Virtual Environment & Setup
- Create and activate a venv:
  - `make venv` then `source .venv/bin/activate`
- Install dependencies (CUDA 11.8 wheels required):
  - `make install` (internally runs `pip install -r requirements.txt` with cu118 wheels)
- Optional dev tools (formatter/linter/tests):
  - `make dev-tools`
- Note: CUDA GPU is required. Runners enforce `torch.cuda.is_available()`.

## Build, Test, and Development Commands
- Smoke test environment: `make test`
- Recommended runners (root-path layout):
  - Unified: `bash scripts/run_experiments.sh`
  - Classification (legacy-style flow):
    - Base: `bash scripts/run_classification_base.sh`
    - PiFi-LAST: `bash scripts/run_classification_pifi_last.sh`
    - PiFi-ILM: `bash scripts/run_classification_pifi_ilm.sh`
  - Entailment (legacy-style flow):
    - Base: `bash scripts/run_entailment_base.sh`
    - PiFi-LAST: `bash scripts/run_entailment_pifi_last.sh`
    - PiFi-ILM: `bash scripts/run_entailment_pifi_ilm.sh`
- Legacy runners (kept): `scripts/run_classification_ilm.sh`, `scripts/run_entailment_ilm.sh` 등
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
- Override paths via CLI flags (`--data_path`, `--model_path`, `--result_path`, etc.). Root-path layout is default in new scripts.
- Do not commit datasets, model weights, credentials, or WANDB tokens. Configure W&B via `.env`.
- `.env` is auto-loaded by runners; if `WANDB_API_KEY` is missing, W&B falls back to offline mode.

## ILM Selection (Reproducibility)
- Default selection settings aligned with legacy behavior when ILM is enabled:
  - `--selection_pooling mean` (masked mean pooling)
  - `--selection_dtype fp16` (GPU half precision)
  - `--selection_max_length 128`
  - Samples/PCs/TopPC: `SEL_SAMPLES=400`, `SEL_PCS=16`, `SEL_TOP_PC=5`
- Selection output path (root layout):
  - `results/layer_selection/<task>/<dataset>/<slm>/<llm>/selection.json`
  - Used to feed `--layer_num` during testing if present.
- Dataset specifics:
  - sst2: SetFit/sst2 preferred; falls back to GLUE sst2 with proper text key.
  - mnli: validation split maps to `validation_matched`.

## Logging
- W&B and TensorBoard are supported. New runners default to off for reproducibility; enable via `USE_WANDB=true USE_TENSORBOARD=true`.
- W&B project/entity can be set via `.env` (`WANDB_PROJECT`, `WANDB_ENTITY`).
- Offline fallback: if no `WANDB_API_KEY`, runners set `WANDB_MODE=offline` automatically.
