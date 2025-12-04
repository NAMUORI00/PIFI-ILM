"""
Unified checkpoint management for PiFi.
Provides consistent save/load functionality with metadata and RNG state preservation.
"""

import os
import random
import argparse
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict, field

import torch
import numpy as np


@dataclass
class CheckpointMetadata:
    """
    Metadata stored with each checkpoint for reproducibility and tracking.
    """
    # Training state
    epoch: int
    wandb_id: Optional[str] = None

    # Experiment configuration (for validation on resume)
    task: str = ""
    dataset: str = ""
    model_type: str = ""
    method: str = ""
    llm_model: str = ""
    layer_num: int = -1
    padding: str = ""
    seed: int = 2023

    # Best metric info
    best_metric: Optional[float] = None
    best_metric_name: Optional[str] = None
    best_epoch_idx: int = 0
    early_stopping_counter: int = 0

    # RNG states for exact reproducibility (populated during save)
    torch_rng_state: Optional[Any] = field(default=None, repr=False)
    numpy_rng_state: Optional[Any] = field(default=None, repr=False)
    python_rng_state: Optional[Any] = field(default=None, repr=False)
    cuda_rng_state: Optional[Any] = field(default=None, repr=False)


def create_metadata_from_args(
    args: argparse.Namespace,
    epoch: int,
    wandb_id: Optional[str] = None,
    best_metric: Optional[float] = None,
    best_metric_name: Optional[str] = None,
    best_epoch_idx: int = 0,
    early_stopping_counter: int = 0
) -> CheckpointMetadata:
    """
    Create CheckpointMetadata from args namespace.

    Args:
        args: Parsed command line arguments
        epoch: Current epoch number
        wandb_id: W&B run ID for resume support
        best_metric: Best validation metric value
        best_metric_name: Name of the metric being optimized
        best_epoch_idx: Epoch index when best metric was achieved
        early_stopping_counter: Early stopping counter at save time

    Returns:
        CheckpointMetadata instance
    """
    return CheckpointMetadata(
        epoch=epoch,
        wandb_id=wandb_id,
        task=getattr(args, 'task', ''),
        dataset=getattr(args, 'task_dataset', ''),
        model_type=getattr(args, 'model_type', ''),
        method=getattr(args, 'method', ''),
        llm_model=getattr(args, 'llm_model', ''),
        layer_num=getattr(args, 'layer_num', -1),
        padding=getattr(args, 'padding', ''),
        seed=getattr(args, 'seed', 2023),
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        best_epoch_idx=best_epoch_idx,
        early_stopping_counter=early_stopping_counter
    )


def save_checkpoint(
    path: str,
    model_state: Dict,
    optimizer_state: Dict,
    scheduler_state: Optional[Dict],
    metadata: CheckpointMetadata
) -> None:
    """
    Save a checkpoint with all necessary state for reproducible resume.

    Args:
        path: Full path to save checkpoint file
        model_state: model.state_dict()
        optimizer_state: optimizer.state_dict()
        scheduler_state: scheduler.state_dict() or None
        metadata: CheckpointMetadata instance
    """
    # Capture current RNG states
    metadata.torch_rng_state = torch.get_rng_state()
    metadata.numpy_rng_state = np.random.get_state()
    metadata.python_rng_state = random.getstate()
    if torch.cuda.is_available():
        metadata.cuda_rng_state = torch.cuda.get_rng_state_all()

    checkpoint = {
        'model': model_state,
        'optimizer': optimizer_state,
        'scheduler': scheduler_state,
        'metadata': asdict(metadata),
        # Legacy fields for backward compatibility
        'epoch': metadata.epoch,
        'wandb_id': metadata.wandb_id
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: str = 'cpu') -> Dict:
    """
    Load checkpoint and return all components.

    Args:
        path: Path to checkpoint file
        map_location: Device to map tensors to

    Returns:
        Dictionary with keys: 'model', 'optimizer', 'scheduler', 'metadata'
        metadata is a CheckpointMetadata instance or None for legacy checkpoints
    """
    checkpoint = torch.load(path, map_location=map_location)

    # Convert metadata dict back to dataclass
    if 'metadata' in checkpoint and checkpoint['metadata'] is not None:
        # Filter out any extra keys that might not be in the dataclass
        meta_dict = checkpoint['metadata']
        valid_fields = {f.name for f in CheckpointMetadata.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in meta_dict.items() if k in valid_fields}
        checkpoint['metadata'] = CheckpointMetadata(**filtered_dict)
    else:
        # Legacy checkpoint without metadata - create minimal metadata
        checkpoint['metadata'] = CheckpointMetadata(
            epoch=checkpoint.get('epoch', 0),
            wandb_id=checkpoint.get('wandb_id')
        )

    return checkpoint


def restore_rng_states(metadata: CheckpointMetadata) -> None:
    """
    Restore RNG states from checkpoint metadata for exact reproducibility.

    Args:
        metadata: CheckpointMetadata with saved RNG states
    """
    if metadata.torch_rng_state is not None:
        torch.set_rng_state(metadata.torch_rng_state)

    if metadata.numpy_rng_state is not None:
        np.random.set_state(metadata.numpy_rng_state)

    if metadata.python_rng_state is not None:
        random.setstate(metadata.python_rng_state)

    if metadata.cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(metadata.cuda_rng_state)


def get_experiment_path(base_path: str, args: argparse.Namespace) -> str:
    """
    Generate a unified experiment path.

    Structure: {base_path}/{task}/{dataset}/{padding}/{model_type}/{method}/{llm_model}/{layer_num}/

    This unifies:
    - Checkpoint save paths
    - Model save paths
    - Resume load paths

    Args:
        base_path: Base directory (e.g., args.checkpoint_path)
        args: Parsed arguments with experiment configuration

    Returns:
        Full experiment directory path
    """
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
    """Get checkpoint directory path"""
    return get_experiment_path(args.checkpoint_path, args)


def get_model_path(args: argparse.Namespace) -> str:
    """Get final model directory path"""
    return get_experiment_path(args.model_path, args)
