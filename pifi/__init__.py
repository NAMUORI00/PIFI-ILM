"""
PiFi (Plugin Fine-tuning) Methodology Module

This module contains the core PiFi methodology components:
- BasePiFiModel: Base model class for PiFi architecture
- layer_loader: LLM layer extraction utilities
- paths: Experiment path management
- logging: PiFi-specific logging (PiFiLogger, create_wandb_config)
"""

from .layer_loader import get_huggingface_model_name, llm_layer
from .paths import (
    get_logs_base,
    get_preprocessed_dir,
    get_experiment_dir,
    get_checkpoint_dir,
    get_model_path,
    # Legacy (deprecated)
    get_experiment_path,
    get_checkpoint_path,
)
from .logging import PiFiLogger, create_wandb_config, get_wandb_exp_name
from .model import BasePiFiModel

__all__ = [
    # Layer loader
    'get_huggingface_model_name',
    'llm_layer',
    # Paths (new unified API)
    'get_logs_base',
    'get_preprocessed_dir',
    'get_experiment_dir',
    'get_checkpoint_dir',
    'get_model_path',
    # Paths (legacy - deprecated)
    'get_experiment_path',
    'get_checkpoint_path',
    # Logging
    'PiFiLogger',
    'create_wandb_config',
    'get_wandb_exp_name',
    # Model
    'BasePiFiModel',
]
