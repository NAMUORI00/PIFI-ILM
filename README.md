https://www.notion.so/namuori00/PIFI-ILM-2a2dcd44779f8014b65bf169b1383d0b?source=copy_link
# PiFi: Plug-in and Fine-tuning

![pifi_figure](https://github.com/user-attachments/assets/e73cbce8-e680-419e-a883-13d05c5e2d98)

## Overview

PiFi is a novel methodology that bridges the gap between Small Language Models (SLMs) and Large Language Models (LLMs) by leveraging the frozen last layer of LLMs as a plug-in component for SLMs during fine-tuning. This approach enables SLMs to benefit from the rich representation capabilities of LLMs while maintaining computational efficiency.

**Paper**: [Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](https://aclanthology.org/2025.acl-long.271/) (ACL 2025)

## Methodology

PiFi works by:
1. **Freezing the Last Layer**: Taking the final layer from a pre-trained LLM and keeping it frozen
2. **Plug-in Architecture**: Integrating this frozen layer as a plug-in component into the SLM architecture
3. **Fine-tuning**: Training the SLM with the plugged-in LLM layer to improve performance on downstream tasks

This approach allows SLMs to leverage the knowledge encoded in LLM's final representations without the computational overhead of running the entire LLM during inference.

## Repository Structure

```
PiFi/
├── main.py                  # Unified entry point for all tasks
├── core/                    # Shared utilities and pipeline
│   ├── arguments.py         # Unified argument parser
│   ├── utils.py             # Common utility functions
│   └── pipeline.py          # Task dispatcher and ILM integration
├── tasks/                   # Task implementations
│   ├── classification/      # Classification tasks (SST-2, IMDB, etc.)
│   └── entailment/          # Textual entailment tasks (MNLI, SNLI)
├── selection/               # ILM layer selection
│   └── ilm_direct.py        # Auto layer selection logic
├── scripts/                 # Experiment scripts
│   ├── run_experiments.sh   # Unified experiment runner
│   ├── run_classification.sh # Classification-only wrapper
│   ├── run_entailment.sh    # Entailment-only wrapper
│   └── README.md            # Detailed script documentation
├── Classification/          # Legacy structure (still functional)
├── TextualEntailment/       # Legacy structure (still functional)
└── requirements.txt         # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PiFi.git
cd PiFi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

The repository provides a unified entry point and experiment scripts:

```bash
# Run all experiments (classification + entailment)
bash scripts/run_experiments.sh

# Run classification only
bash scripts/run_classification.sh

# Run entailment only
bash scripts/run_entailment.sh

# Run with ILM auto layer selection
bash scripts/run_with_ilm.sh
```

### Unified Main Script

All tasks can be run using the unified `main.py`:

```bash
# Classification
python main.py --task classification --job training --task_dataset sst2 --method pifi

# Entailment
python main.py --task entailment --job training --task_dataset mnli --method pifi

# With ILM auto-selection
python main.py --task classification --job training --task_dataset sst2 \
    --method pifi --llm_model llama3.1 --auto_select_layer true
```

## Docker (GPU)

This repository includes a Dockerfile and optional docker-compose for convenience.

Build the image:
```bash
docker build -t pifi:cu118 .
```

Interactive shell with mounted artifacts:
```bash
docker run --rm -it --gpus all \
  -v $PWD/cache:/app/cache \
  -v $PWD/preprocessed:/app/preprocessed \
  -v $PWD/models:/app/models \
  -v $PWD/checkpoints:/app/checkpoints \
  -v $PWD/results:/app/results \
  -v $PWD/tensorboard_logs:/app/tensorboard_logs \
  -v $PWD/wandb:/app/wandb \
  -v $PWD/.hf_cache:/opt/hf-cache \
  -v $PWD/dataset:/app/dataset \
  -w /app \
  pifi:cu118 bash
```

One-shot training example:
```bash
docker run --rm --gpus all \
  -v $PWD/cache:/app/cache -v $PWD/preprocessed:/app/preprocessed \
  -v $PWD/models:/app/models -v $PWD/checkpoints:/app/checkpoints \
  -v $PWD/results:/app/results -v $PWD/tensorboard_logs:/app/tensorboard_logs \
  -w /app --entrypoint python pifi:cu118 \
  main.py --task classification --job training --task_dataset sst2 --method base
```

### Docker Compose (optional)

Build and keep container running, then exec commands:
```bash
docker compose build
docker compose up -d pifi
# shell inside container
docker compose exec pifi bash
```

First-run auto experiment (classification + ILM):
```bash
# The container entrypoint will automatically run a quick classification + ILM run on first start
# (defaults: DATASETS=sst2, MODEL=bert, LLM=llama3.1, EPOCHS=1, BS=8, USE_WANDB=false, USE_TENSORBOARD=false)
# A marker is written to results/.first_run_done to avoid re-running next time.

docker compose down
rm -f results/.first_run_done  # optional: force re-run
docker compose up -d pifi

# watch logs
docker compose logs -f pifi

# to skip auto-run
FIRST_RUN=false docker compose up -d pifi
```

Compose quick test flow (GPU + HF cache path set):
```bash
# 0) GPU check
docker compose exec pifi python -c "import torch; print('CUDA', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 1) Preprocess (sst2)
docker compose exec -e HF_HOME=/app/cache/hf \
  pifi python main.py --task classification --job preprocessing --task_dataset sst2 \
  --use_wandb false --num_workers 2 --max_seq_len 64 --train_valid_split 0.95

# 2) Train base (1 epoch)
docker compose exec -e HF_HOME=/app/cache/hf \
  pifi python main.py --task classification --job training --task_dataset sst2 --method base \
  --num_epochs 1 --batch_size 16 --use_wandb false --num_workers 2 --max_seq_len 64

# 3) Test
docker compose exec -e HF_HOME=/app/cache/hf \
  pifi python main.py --task classification --job testing --task_dataset sst2 --method base \
  --batch_size 32 --use_wandb false --num_workers 2 --max_seq_len 64

# 4) Stop services
docker compose down
```

Notes:
- Some Docker/Compose versions may ignore GPU reservations; if so, use `docker compose run --gpus all ...` or the `up`+`exec` sequence above.
- The entrypoint sets `HF_HOME=/app/cache/hf` (and `TRANSFORMERS_CACHE`/`HF_DATASETS_CACHE`) to avoid cache permission issues.
- You can override first-run parameters via env vars on `up`: `DATASETS`, `MODEL`, `LLM`, `EPOCHS`, `BS`, `WORKERS`, `USE_WANDB`.

### Classification Tasks

The classification module supports multiple datasets: SST-2, IMDB, Tweet Sentiment Binary, Tweet Offensive, and CoLA.

#### Running Classification Experiments

**Using the unified script (recommended):**
```bash
# Run all classification datasets with ILM
TASKS="classification" USE_ILM=true bash scripts/run_experiments.sh

# Run specific datasets
TASKS="classification" DATASETS="sst2 imdb" bash scripts/run_experiments.sh

# Custom configuration
LLM=llama3.1 MODEL=roberta EPOCHS=5 BS=16 \
TASKS="classification" bash scripts/run_experiments.sh
```

**Using main.py directly:**
```bash
# Preprocessing
python main.py --task classification --job preprocessing --task_dataset sst2

# Training baseline model
python main.py --task classification --job training --task_dataset sst2 --method base

# Testing baseline model
python main.py --task classification --job testing --task_dataset sst2 --method base

# Training with PiFi + ILM auto-selection
python main.py --task classification --job training --task_dataset sst2 \
    --method pifi --llm_model llama3.1 --auto_select_layer true

# Testing with PiFi
python main.py --task classification --job testing --task_dataset sst2 \
    --method pifi --llm_model llama3.1
```

#### Available Parameters:
- `--task_dataset`: Dataset name (sst2, imdb, tweet_sentiment_binary, tweet_offensive, cola)
- `--model_type`: Base model type (bert, roberta, albert, electra, deberta, debertav3)
- `--method`: Training method (base, pifi)
- `--llm_model`: LLM to use for plugin (llama3.1, mistral0.1, mistral0.3, qwen2_7b, gemma2, falcon)
- `--job`: Operation to perform (preprocessing, training, testing)
 - `--auto_select_layer`: Automatically select LLM layer (true/false)
 - `--selection_samples`, `--selection_pcs`, `--selection_top_pc`: Selection controls

### Textual Entailment Tasks

The textual entailment module supports MNLI and SNLI datasets.

#### Running Entailment Experiments

**Using the unified script (recommended):**
```bash
# Run all entailment datasets with ILM
TASKS="entailment" USE_ILM=true bash scripts/run_experiments.sh

# Run specific datasets
TASKS="entailment" DATASETS="mnli snli" bash scripts/run_experiments.sh

# Custom configuration
LLM=llama3.1 MODEL=roberta EPOCHS=5 \
TASKS="entailment" bash scripts/run_experiments.sh
```

**Using main.py directly:**
```bash
# Preprocessing
python main.py --task entailment --job preprocessing --task_dataset mnli

# Training baseline model
python main.py --task entailment --job training --task_dataset mnli --method base

# Testing baseline model
python main.py --task entailment --job testing --task_dataset mnli --method base

# Training with PiFi + ILM auto-selection
python main.py --task entailment --job training --task_dataset mnli \
    --method pifi --llm_model llama3.1 --auto_select_layer true

# Testing with PiFi
python main.py --task entailment --job testing --task_dataset mnli \
    --method pifi --llm_model llama3.1
```

#### Available Parameters:
- `--task_dataset`: Dataset name (mnli, snli)
- `--model_type`: Base model type (bert, roberta, albert, electra, deberta, debertav3)
- `--method`: Training method (base, pifi)
- `--llm_model`: LLM to use for plugin (llama3.1, mistral0.1, mistral0.3, qwen2_7b, gemma2, falcon)
- `--job`: Operation to perform (preprocessing, training, testing)
 - `--auto_select_layer`: Automatically select LLM layer (true/false)
 - `--selection_samples`, `--selection_pcs`, `--selection_top_pc`: Selection controls

## Experimental Setup

The experiments compare two approaches:
1. **Baseline (`base`)**: Standard SLM fine-tuning
2. **PiFi (`pifi`)**: SLM fine-tuning with frozen LLM layer plugin

Both approaches are evaluated on the same downstream tasks to demonstrate the effectiveness of the PiFi methodology.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kim-etal-2025-plug,
    title = "Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models",
    author = "Kim, Kyeonghyun  and
      Jang, Jinhee  and
      Choi, Juhwan  and
      Lee, Yoonji  and
      Jin, Kyohoon  and
      Kim, YoungBin",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.271/",
    doi = "10.18653/v1/2025.acl-long.271",
    pages = "5434--5452",
    ISBN = "979-8-89176-251-0",
    abstract = "Large language models (LLMs) are renowned for their extensive linguistic knowledge and strong generalization capabilities, but their high computational demands make them unsuitable for resource-constrained environments. In contrast, small language models (SLMs) are computationally efficient but often lack the broad generalization capacity of LLMs. To bridge this gap, we propose PiFi, a novel framework that combines the strengths of both LLMs and SLMs to achieve high performance while maintaining efficiency. PiFi integrates a single frozen layer from an LLM into a SLM and fine-tunes the combined model for specific tasks, boosting performance without a significant increase in computational cost. We show that PiFi delivers consistent performance improvements across a range of natural language processing tasks, including both natural language understanding and generation. Moreover, our findings demonstrate PiFi{'}s ability to effectively leverage LLM knowledge, enhancing generalization to unseen domains and facilitating the transfer of linguistic abilities."
}
```
