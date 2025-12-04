"""
PiFi-Specific Logging Utilities

This module provides:
- PiFiLogger: Unified logging for both W&B and local files
- create_wandb_config: Factory function for W&B configuration
- get_wandb_exp_name: Experiment name generation
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd


def get_wandb_exp_name(args: argparse.Namespace) -> str:
    """
    Get the experiment name for Weights & Biases experiment.

    Args:
        args: Parsed command line arguments

    Returns:
        Formatted experiment name string
    """
    exp_name = str(args.seed)
    exp_name += " %s - " % args.task.upper()
    exp_name += "%s / " % args.task_dataset.upper()
    if args.task_dataset != args.test_dataset:
        exp_name += "%s / " % args.test_dataset
    exp_name += "%s" % args.model_type.upper()
    exp_name += " - %s" % args.method.upper()
    if 'ablation' in args.proj_name:
        exp_name += ' - %s' % args.padding.upper()
    if args.method == 'pifi':
        if args.freeze == False:
            exp_name += " - freeze(x)"
        exp_name += " - %s" % args.llm_model.upper()
        exp_name += "(%s)" % args.layer_num

    return exp_name


class LocalFallbackLogger:
    """
    Fallback logger that saves to local files when W&B is disabled.
    Saves to unified experiment directory.
    """

    def __init__(self, experiment_dir: str):
        """
        Initialize with experiment directory.

        Args:
            experiment_dir: Full path to experiment directory
                (from get_experiment_dir() or manually constructed)
        """
        self.base_path = experiment_dir
        os.makedirs(self.base_path, exist_ok=True)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'LocalFallbackLogger':
        """Create logger from parsed arguments using unified path structure."""
        from .paths import get_experiment_dir
        return cls(get_experiment_dir(args))

    def log_json(self, data: Dict[str, Any], filename: str) -> str:
        """Save data as JSON file, return path."""
        filepath = os.path.join(self.base_path, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def log_figure(self, figure: plt.Figure, filename: str, dpi: int = 150) -> str:
        """Save matplotlib figure to PNG, return path."""
        filepath = os.path.join(self.base_path, filename)
        figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
        return filepath

    def log_table(self, dataframe: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to CSV, return path."""
        filepath = os.path.join(self.base_path, filename)
        dataframe.to_csv(filepath, index=False)
        return filepath


class PiFiLogger:
    """
    Unified logging facade - handles both W&B and local file logging.
    When W&B is enabled: logs to both W&B and local files.
    When W&B is disabled: logs to local files only.
    """

    def __init__(self, use_wandb: bool, experiment_dir: str,
                 result_path: str = '', task: str = '', dataset: str = '',
                 model_type: str = '', method: str = '', llm_model: str = '', layer_num: int = -1):
        """
        Initialize PiFiLogger.

        Args:
            use_wandb: Whether to use W&B logging
            experiment_dir: Full path to experiment directory (from get_experiment_dir())
            result_path: Result path (for legacy selection path)
            task, dataset, model_type, method, llm_model, layer_num: Experiment metadata
        """
        self._use_wandb = use_wandb
        self._experiment_dir = experiment_dir
        self._result_path = result_path
        self._task = task
        self._dataset = dataset
        self._model_type = model_type
        self._method = method
        self._llm_model = llm_model
        self._layer_num = layer_num
        self._local_logger = LocalFallbackLogger(experiment_dir)
        # Lazy import WandbManager to avoid circular imports
        self._wandb_mgr = None
        if use_wandb:
            from core.wandb_manager import WandbManager
            self._wandb_mgr = WandbManager.get_instance()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'PiFiLogger':
        """Create logger from parsed arguments using unified path structure."""
        from .paths import get_experiment_dir
        return cls(
            use_wandb=getattr(args, 'use_wandb', False),
            experiment_dir=get_experiment_dir(args),
            result_path=getattr(args, 'result_path', 'results'),
            task=getattr(args, 'task', 'classification'),
            dataset=getattr(args, 'task_dataset', 'sst2'),
            model_type=getattr(args, 'model_type', 'bert'),
            method=getattr(args, 'method', 'pifi'),
            llm_model=getattr(args, 'llm_model', ''),
            layer_num=getattr(args, 'layer_num', -1)
        )

    @property
    def local_path(self) -> str:
        """Get the local logging directory path."""
        return self._local_logger.base_path

    def log_selection(self, effects: List[float], best_layer: int,
                      metadata: Dict[str, Any],
                      figures: Optional[Dict[str, plt.Figure]] = None) -> str:
        """
        Log ILM selection results to W&B and local files.

        Args:
            effects: List of effect scores per layer
            best_layer: Selected best layer index
            metadata: Additional metadata (task, dataset, etc.)
            figures: Optional dict of figure name -> matplotlib Figure

        Returns:
            Path to saved selection.json
        """
        # Prepare selection data
        selection_data = {
            **metadata,
            'effects': effects,
            'best_llm_layer': best_layer,
            'timestamp': datetime.now().isoformat()
        }

        # Always save locally
        json_path = self._local_logger.log_json(selection_data, 'selection.json')

        # Mirror to historical layer_selection path for downstream consumers
        if self._method == 'pifi' and self._llm_model:
            legacy_json_path = os.path.join(
                self._result_path,
                'layer_selection',
                self._task,
                self._dataset,
                self._model_type,
                self._llm_model,
                'selection.json'
            )
            try:
                os.makedirs(os.path.dirname(legacy_json_path), exist_ok=True)
                with open(legacy_json_path, 'w') as f:
                    json.dump(selection_data, f, indent=2, default=str)
                json_path = legacy_json_path
            except Exception as e:
                print(f"[selection] Failed to write legacy selection file: {e}")

        # Save figures locally
        if figures:
            for name, fig in figures.items():
                self._local_logger.log_figure(fig, f'{name}.png')

        # Log to W&B if enabled
        if self._use_wandb and self._wandb_mgr and self._wandb_mgr.is_initialized:
            import wandb
            # Log effects as table
            effects_df = pd.DataFrame({
                'layer': list(range(len(effects))),
                'effect': effects
            })
            self._wandb_mgr.log_table('selection/effects_table', effects_df)

            # Log figures to W&B
            if figures:
                for name, fig in figures.items():
                    self._wandb_mgr.log_figure(f'selection/{name}', fig)

            # Log best layer as summary
            self._wandb_mgr.log_summary({
                'best_llm_layer': best_layer,
                'best_layer_effect': effects[best_layer] if best_layer >= 0 else None
            })

            # Log selection.json as artifact
            artifact = wandb.Artifact(
                f"selection-{metadata.get('dataset', 'unknown')}",
                type='selection'
            )
            artifact.add_file(json_path)
            wandb.log_artifact(artifact)

        return json_path

    def log_test_results(self, metrics: Dict[str, float],
                         metadata: Dict[str, Any]) -> str:
        """
        Log test results to W&B and local files.

        Args:
            metrics: Test metrics (accuracy, f1, loss, etc.)
            metadata: Additional metadata (task, dataset, config, etc.)

        Returns:
            Path to saved test_results.json
        """
        # Prepare test result data
        test_data = {
            **metadata,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Always save locally
        json_path = self._local_logger.log_json(test_data, 'test_results.json')

        # Log to W&B if enabled
        if self._use_wandb and self._wandb_mgr and self._wandb_mgr.is_initialized:
            # Log as table
            results_df = pd.DataFrame([{
                'Dataset': metadata.get('dataset', ''),
                'Model': metadata.get('model_type', ''),
                'Method': metadata.get('method', ''),
                **metrics
            }])
            self._wandb_mgr.log_table('TEST_Result', results_df)

            # Log individual metrics
            for key, value in metrics.items():
                self._wandb_mgr.log({f'TEST/{key}': value})

            # Log summary
            self._wandb_mgr.log_summary({
                'test_accuracy': metrics.get('accuracy', metrics.get('acc')),
                'test_f1': metrics.get('f1'),
                'test_loss': metrics.get('loss')
            })

        return json_path

    def log_training_summary(self, best_epoch: int, best_metric: float,
                             best_metric_name: str, total_epochs: int) -> None:
        """
        Log training summary at end of training.

        Args:
            best_epoch: Epoch with best validation metric
            best_metric: Best validation metric value
            best_metric_name: Name of the metric being optimized
            total_epochs: Total number of epochs trained
        """
        summary_data = {
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'best_metric_name': best_metric_name,
            'total_epochs': total_epochs,
            'timestamp': datetime.now().isoformat()
        }

        # Save locally
        self._local_logger.log_json(summary_data, 'training_summary.json')

        # Log to W&B if enabled
        if self._use_wandb and self._wandb_mgr and self._wandb_mgr.is_initialized:
            self._wandb_mgr.log_summary({
                'best_epoch': best_epoch,
                f'best_{best_metric_name}': best_metric,
                'total_epochs': total_epochs
            })


def create_wandb_config(args: argparse.Namespace, job_type: str = 'train'):
    """
    Factory function to create WandbConfig from args.
    Consolidates experiment naming logic in one place.

    Args:
        args: Parsed command line arguments
        job_type: Type of job ('train', 'test', 'selection')

    Returns:
        WandbConfig instance ready for initialization
    """
    from core.wandb_manager import WandbConfig

    project = os.environ.get('WANDB_PROJECT') or args.proj_name
    entity = os.environ.get('WANDB_ENTITY')

    # Build experiment name
    name = f"{args.seed} {args.task.upper()} - {args.task_dataset.upper()}"
    if args.task_dataset != args.test_dataset:
        name += f" / {args.test_dataset}"
    name += f" / {args.model_type.upper()} - {args.method.upper()}"

    if 'ablation' in args.proj_name:
        name += f' - {args.padding.upper()}'

    if args.method == 'pifi':
        if not getattr(args, 'freeze', True):
            name += " - freeze(x)"
        name += f" - {args.llm_model.upper()}({args.layer_num})"

    if job_type == 'test':
        name += ' - Test'
    elif job_type == 'selection':
        name += ' - Selection'

    # Build tags
    tags = [
        job_type.upper(),
        f"Dataset: {args.task_dataset}",
        f"Model: {args.model_type}",
        f"Method: {args.method}",
    ]
    if args.method == 'pifi':
        tags.extend([
            f"LLM: {args.llm_model}",
            f"LLM_Layer: {args.layer_num}"
        ])

    return WandbConfig(
        project=project,
        name=name,
        entity=entity,
        config=vars(args),
        notes=getattr(args, 'description', ''),
        tags=tags
    )
