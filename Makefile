SHELL := /bin/bash

# Python virtual environment
PYTHON ?= python3
VENV_DIR ?= .venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# PyTorch wheel source (defaults to CUDA 11.8). Use `make install-cpu` for CPU-only.
TORCH_CUDA ?= cu118
TORCH_INDEX ?= https://download.pytorch.org/whl/$(TORCH_CUDA)

.PHONY: venv install dev-tools classify entail test clean

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

install: venv
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

dev-tools:
	$(PIP) install black ruff pytest

# Run tasks (pass args via ARGS="--job=... --task_dataset=..." )
classify:
	cd Classification && $(PY) main.py $(ARGS)

entail:
	cd TextualEntailment && $(PY) main.py $(ARGS)

# Lightweight environment + CUDA smoke test
test:
	$(PY) -c "import sys, torch, transformers; print('Python:', sys.version.split()[0]); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Transformers:', transformers.__version__)"

clean:
	rm -rf $(VENV_DIR)
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete -o -name "*.pyo" -delete
