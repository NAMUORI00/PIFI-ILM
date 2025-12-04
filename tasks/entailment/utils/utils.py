"""
Entailment Utils - Wrapper for core.utils
This maintains backward compatibility with existing imports
"""

# Import everything from core.utils
from core.utils import (
    check_path,
    set_random_seed,
    get_torch_device,
    TqdmLoggingHandler,
    write_log,
    get_wandb_exp_name,
    get_huggingface_model_name,
    llm_layer,
    worker_init_fn,
)

# Re-export parse_bool from core.arguments for backward compatibility
from core.arguments import parse_bool

# Expose all functions
__all__ = [
    'check_path',
    'set_random_seed',
    'get_torch_device',
    'TqdmLoggingHandler',
    'write_log',
    'get_wandb_exp_name',
    'get_huggingface_model_name',
    'llm_layer',
    'worker_init_fn',
    'parse_bool',
]
