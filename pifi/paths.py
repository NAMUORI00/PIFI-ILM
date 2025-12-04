"""
Experiment Path Management for PiFi

This module provides functions to generate consistent experiment paths
for checkpoints, models, and results across all PiFi experiments.

Unified Path Structure (v3):
    {logs_path}/
    ├── preprocessed/                 # Preprocessed data
    │   └── {task}/{dataset}/{model_type}/
    │       ├── train_processed.pkl
    │       ├── valid_processed.pkl
    │       └── test_processed.pkl
    │
    └── experiments/                  # Experiment artifacts
        └── {task}/{dataset}/{padding}/{model_type}/{method}/{llm_model}/{layer_num}/
            ├── checkpoints/
            │   ├── checkpoint.pt   # best model checkpoint
            │   └── last.pt         # latest checkpoint
            ├── final_model.pt      # final trained model
            ├── selection.json      # ILM layer selection results
            ├── test_results.json   # test metrics
            └── training_summary.json  # training summary
"""

import os
import argparse
import warnings


def get_logs_base(args: argparse.Namespace) -> str:
    """
    Get the logs base directory path.

    Supports both new (logs_path) and legacy (result_path) arguments.

    Args:
        args: Parsed arguments

    Returns:
        Base logs directory path
    """
    # Prefer logs_path if available, fall back to result_path
    return getattr(args, 'logs_path', None) or getattr(args, 'result_path', 'logs')


def get_preprocessed_dir(args: argparse.Namespace, dataset: str = None) -> str:
    """
    Get preprocessed data directory path.

    Structure: {logs}/preprocessed/{task}/{dataset}/{model_type}/

    Args:
        args: Parsed arguments with task, task_dataset, model_type
        dataset: Optional dataset name override (e.g., args.test_dataset for testing)

    Returns:
        Full preprocessed data directory path
    """
    return os.path.join(
        get_logs_base(args),
        'preprocessed',
        args.task,
        dataset or args.task_dataset,
        args.model_type
    )


def get_experiment_dir(args: argparse.Namespace) -> str:
    """
    Get unified experiment directory path.

    All experiment artifacts (checkpoints, models, logs) are stored under this directory.

    Structure: {logs}/experiments/{task}/{dataset}/{padding}/{model_type}/{method}/{llm_model}/{layer_num}/

    Args:
        args: Parsed arguments with logs_path/result_path and experiment config

    Returns:
        Full experiment directory path
    """
    path_parts = [
        get_logs_base(args),
        'experiments',
        args.task,
        args.task_dataset,
        args.padding,
        args.model_type,
        args.method,
    ]

    # Add LLM-specific paths for pifi method
    if args.method == 'pifi':
        path_parts.extend([args.llm_model, str(args.layer_num)])
    else:
        # For non-pifi, use 'none' placeholders for consistent structure
        path_parts.extend(['none', 'none'])

    return os.path.join(*path_parts)


def get_checkpoint_dir(args: argparse.Namespace) -> str:
    """
    Get checkpoint directory path.

    Structure: {experiment_dir}/checkpoints/

    Args:
        args: Parsed arguments with experiment config

    Returns:
        Full checkpoint directory path
    """
    return os.path.join(get_experiment_dir(args), 'checkpoints')


def get_model_path(args: argparse.Namespace) -> str:
    """
    Get final model file path.

    Structure: {experiment_dir}/final_model.pt

    Args:
        args: Parsed arguments with experiment config

    Returns:
        Full path to final_model.pt
    """
    return os.path.join(get_experiment_dir(args), 'final_model.pt')


# =============================================================================
# Legacy API (Deprecated - for backward compatibility)
# =============================================================================

def get_experiment_path(base_path: str, args: argparse.Namespace) -> str:
    """
    [DEPRECATED] Generate a unified experiment path.

    This function is deprecated. Use get_experiment_dir() instead.

    Structure: {base_path}/{task}/{dataset}/{padding}/{model_type}/{method}/{llm_model}/{layer_num}/

    Args:
        base_path: Base directory (e.g., args.checkpoint_path)
        args: Parsed arguments with experiment configuration

    Returns:
        Full experiment directory path
    """
    warnings.warn(
        "get_experiment_path() is deprecated. Use get_experiment_dir() for unified paths.",
        DeprecationWarning,
        stacklevel=2
    )
    path_parts = [
        base_path,
        args.task,
        args.task_dataset,
        args.padding,
        args.model_type,
        args.method,
    ]

    # Add LLM-specific paths for pifi method
    if args.method == 'pifi':
        path_parts.extend([args.llm_model, str(args.layer_num)])
    else:
        # For non-pifi, use 'none' placeholders for consistent structure
        path_parts.extend(['none', 'none'])

    return os.path.join(*path_parts)


def get_checkpoint_path(args: argparse.Namespace) -> str:
    """
    [DEPRECATED] Get checkpoint directory path using old structure.

    This function is deprecated. Use get_checkpoint_dir() instead.

    Args:
        args: Parsed arguments with checkpoint_path and experiment config

    Returns:
        Full checkpoint directory path
    """
    warnings.warn(
        "get_checkpoint_path() is deprecated. Use get_checkpoint_dir() for unified paths.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_experiment_path(args.checkpoint_path, args)
