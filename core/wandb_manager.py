"""
Centralized W&B (Weights & Biases) management for PiFi
Provides a unified interface for experiment tracking, logging, and resume support.
"""

import os
import subprocess
import argparse
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class WandbConfig:
    """Configuration container for W&B initialization"""
    project: str
    name: str
    entity: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    resume: bool = False
    run_id: Optional[str] = None


class WandbManager:
    """
    Singleton manager for W&B experiment tracking.
    Handles initialization, logging, resume, and cleanup.
    """

    _instance: Optional['WandbManager'] = None

    def __init__(self):
        self._initialized = False
        self._run = None
        self._run_id: Optional[str] = None

    @classmethod
    def get_instance(cls) -> 'WandbManager':
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing or re-initialization)"""
        if cls._instance is not None:
            cls._instance.finish()
        cls._instance = None

    def init(self, config: WandbConfig, model=None, criterion=None) -> str:
        """
        Initialize W&B run.

        Args:
            config: WandbConfig instance with run settings
            model: Optional model to watch for gradient logging
            criterion: Optional loss function for model watching

        Returns:
            run_id: The W&B run ID for checkpoint storage
        """
        import wandb

        init_kwargs = {
            'project': config.project,
            'name': config.name,
            'config': config.config,
            'notes': config.notes,
            'tags': config.tags,
            'settings': wandb.Settings(save_code=True),
        }

        if config.entity:
            init_kwargs['entity'] = config.entity

        if config.resume and config.run_id:
            init_kwargs['resume'] = True
            init_kwargs['id'] = config.run_id

        self._run = wandb.init(**init_kwargs)
        self._run_id = self._run.id
        self._initialized = True

        # Log git commit info
        self._log_git_info()

        # Watch model if provided
        if model is not None:
            wandb.watch(models=model, criterion=criterion, log='all', log_freq=10)

        return self._run_id

    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID"""
        return self._run_id

    @property
    def is_initialized(self) -> bool:
        """Check if W&B is initialized"""
        return self._initialized

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for x-axis
        """
        if not self._initialized:
            return
        import wandb
        wandb.log(metrics, step=step)

    def log_table(self, key: str, dataframe) -> None:
        """
        Log a pandas DataFrame as a W&B table.

        Args:
            key: Table name in W&B
            dataframe: pandas DataFrame to log
        """
        if not self._initialized:
            return
        import wandb
        table = wandb.Table(dataframe=dataframe)
        wandb.log({key: table})

    def log_image(self, key: str, images: List, captions: Optional[List[str]] = None) -> None:
        """
        Log images to W&B.

        Args:
            key: Image group name
            images: List of images (PIL, numpy, or file paths)
            captions: Optional list of captions for each image
        """
        if not self._initialized:
            return
        import wandb
        if captions:
            wandb_imgs = [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]
        else:
            wandb_imgs = [wandb.Image(img) for img in images]
        wandb.log({key: wandb_imgs})

    def log_artifact(self, name: str, artifact_type: str, file_path: str,
                     metadata: Optional[Dict] = None) -> None:
        """
        Log a file as a W&B artifact.

        Args:
            name: Artifact name
            artifact_type: Type of artifact (e.g., 'model', 'dataset')
            file_path: Path to file to upload
            metadata: Optional metadata dictionary
        """
        if not self._initialized:
            return
        import wandb
        artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

    def alert(self, title: str, text: str, level: str = 'INFO',
              wait_duration: int = 300) -> None:
        """
        Send an alert notification.

        Args:
            title: Alert title
            text: Alert message body
            level: Alert level ('INFO', 'WARN', 'ERROR')
            wait_duration: Minimum seconds between alerts of same title
        """
        if not self._initialized:
            return
        import wandb
        from wandb import AlertLevel
        level_map = {
            'INFO': AlertLevel.INFO,
            'WARN': AlertLevel.WARN,
            'ERROR': AlertLevel.ERROR
        }
        wandb.alert(
            title=title,
            text=text,
            level=level_map.get(level, AlertLevel.INFO),
            wait_duration=wait_duration
        )

    def finish(self) -> None:
        """Finish the W&B run and cleanup"""
        if self._initialized:
            import wandb
            wandb.finish()
            self._initialized = False
            self._run = None

    def _log_git_info(self) -> None:
        """Log git commit information to W&B"""
        if not self._initialized:
            return
        import wandb
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            wandb.config.update({'git_commit': commit}, allow_val_change=True)
            short = commit[:7]
            current_tags = list(wandb.run.tags or [])
            wandb.run.tags = list(set(current_tags + [f"GIT:{short}"]))
            wandb.log_code(root=os.getcwd())
        except Exception:
            pass  # Git not available or not a git repo


def create_wandb_config(args: argparse.Namespace, job_type: str = 'train') -> WandbConfig:
    """
    Factory function to create WandbConfig from args.
    Consolidates experiment naming logic in one place.

    Args:
        args: Parsed command line arguments
        job_type: Type of job ('train', 'test', 'selection')

    Returns:
        WandbConfig instance ready for initialization
    """
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
