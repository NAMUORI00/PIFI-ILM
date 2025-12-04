"""
PiFi Core Module
Shared utilities, arguments, and pipeline logic for all tasks.
"""

__all__ = [
    # Arguments
    'ArgParser',
    # Utils
    'check_path',
    'set_random_seed',
    'worker_init_fn',
    'get_torch_device',
    'TqdmLoggingHandler',
    'write_log',
    'get_wandb_exp_name',
    'get_huggingface_model_name',
    'llm_layer',
    # Pipeline
    'run_pipeline',
    # Optimizer & Scheduler
    'get_optimizer',
    'get_scheduler',
    # Checkpoint
    'CheckpointMetadata',
    'save_checkpoint',
    'load_checkpoint',
    'restore_rng_states',
    # Paths (new unified API from pifi.paths)
    'get_logs_base',
    'get_preprocessed_dir',
    'get_experiment_dir',
    'get_checkpoint_dir',
    'get_model_path',
    # Paths (legacy - deprecated, re-exported from pifi.paths)
    'get_experiment_path',
    'get_checkpoint_path',
    # Wandb
    'WandbConfig',
    'WandbManager',
    'PiFiLogger',
    'create_wandb_config',
]

from .arguments import ArgParser
from .utils import (
    check_path,
    set_random_seed,
    worker_init_fn,
    get_torch_device,
    TqdmLoggingHandler,
    write_log,
    get_wandb_exp_name,
    get_huggingface_model_name,
    llm_layer,
)
from .pipeline import run_pipeline
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .checkpoint import (
    CheckpointMetadata,
    save_checkpoint,
    load_checkpoint,
    restore_rng_states,
)
# Import path functions from pifi.paths (unified API)
from pifi.paths import (
    get_logs_base,
    get_preprocessed_dir,
    get_experiment_dir,
    get_checkpoint_dir,
    get_model_path,
    # Legacy (deprecated) - re-exported for backward compatibility
    get_experiment_path,
    get_checkpoint_path,
)
from .wandb_manager import (
    WandbConfig,
    WandbManager,
    PiFiLogger,
    create_wandb_config,
)
