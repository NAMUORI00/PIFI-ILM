SHELL := /bin/bash

# Python virtual environment
PYTHON ?= python3
VENV_DIR ?= .venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# PyTorch wheel source (defaults to CUDA 11.8). Use `make install-cpu` for CPU-only.
TORCH_CUDA ?= cu118
TORCH_INDEX ?= https://download.pytorch.org/whl/$(TORCH_CUDA)

.PHONY: venv install dev-tools classify entail test clean docker-build docker-run docker-test compose-build compose-run compose-shell

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
	$(PY) main.py --task classification $(ARGS)

entail:
	$(PY) main.py --task entailment $(ARGS)

# Lightweight environment + CUDA smoke test
test:
	$(PY) -c "import sys, torch, transformers; print('Python:', sys.version.split()[0]); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Transformers:', transformers.__version__)"

clean:
	rm -rf $(VENV_DIR)
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete -o -name "*.pyo" -delete

# Docker helpers
docker-build:
	docker build -t pifi:cu118 .

docker-run:
	# GPU 필요 시 --gpus all 추가
	docker run --rm -it --gpus all \
	  -e WANDB_MODE=$${WANDB_MODE:-offline} \
	  -v $(PWD)/cache:/app/cache \
	  -v $(PWD)/preprocessed:/app/preprocessed \
	  -v $(PWD)/models:/app/models \
	  -v $(PWD)/checkpoints:/app/checkpoints \
	  -v $(PWD)/results:/app/results \
	  -v $(PWD)/tensorboard_logs:/app/tensorboard_logs \
	  -v $(PWD)/wandb:/app/wandb \
	  -v $(PWD)/.hf_cache:/opt/hf-cache \
	  -v $(PWD)/dataset:/app/dataset \
	  -w /app \
	  pifi:cu118 bash

docker-test:
	docker run --rm --gpus all pifi:cu118 \
	  python -c "import sys, torch, transformers; print('Python:', sys.version.split()[0]); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Transformers:', transformers.__version__)"

# Compose helpers
compose-build:
	docker compose build

compose-run:
	# 원샷 실행 (GPU는 --gpus all 로 전달)
	docker compose run --rm --gpus all pifi python main.py --help

compose-shell:
	docker compose run --rm --gpus all pifi
