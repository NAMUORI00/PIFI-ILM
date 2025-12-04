"""
Base Trainer for PiFi Tasks

Provides common training and validation logic that can be used by
classification and entailment tasks.
"""

import logging
from typing import Dict, Any, Optional, Callable
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from core import write_log


class BaseTrainer:
    """
    Base trainer providing common training and validation logic.

    This class extracts the shared training loop logic from task-specific
    train.py files. It handles:
    - Training one epoch
    - Validation
    - Metrics computation (accuracy, F1, precision, recall)
    - Logging during training

    Usage:
        trainer = BaseTrainer(args, model, device, logger)
        train_metrics = trainer.train_epoch(train_loader, optimizer, scheduler, loss_fn)
        valid_metrics = trainer.validate(valid_loader, loss_fn)
    """

    def __init__(
        self,
        args,
        model: nn.Module,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the trainer.

        Args:
            args: Parsed command line arguments
            model: The model to train
            device: Device to run training on
            logger: Optional logger for output
        """
        self.args = args
        self.model = model
        self.device = device
        self.logger = logger

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: nn.Module,
        epoch_idx: int = 0
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            loss_fn: Loss function
            epoch_idx: Current epoch index (for logging)

        Returns:
            Dictionary of training metrics (loss, accuracy, f1)
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        num_batches = len(dataloader)

        for iter_idx, data_dicts in enumerate(tqdm(
            dataloader,
            total=num_batches,
            desc=f'Training - Epoch [{epoch_idx}/{self.args.num_epochs}]',
            position=0,
            leave=True
        )):
            # Move data to device
            input_ids = data_dicts['input_ids'].to(self.device)
            attention_mask = data_dicts['attention_mask'].to(self.device)
            token_type_ids = data_dicts['token_type_ids'].to(self.device)
            labels = data_dicts['labels'].to(self.device)

            # Forward pass
            logits = self.model(input_ids, attention_mask, token_type_ids)

            # Compute loss and metrics
            batch_loss = loss_fn(logits, labels)
            batch_acc = (logits.argmax(dim=-1) == labels).float().mean()
            batch_f1 = f1_score(
                labels.cpu().numpy(),
                logits.argmax(dim=-1).cpu().numpy(),
                average='macro'
            )

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()

            if self.args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

            optimizer.step()

            # Step scheduler if per-batch
            if scheduler is not None and self.args.scheduler in [
                'StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'
            ]:
                scheduler.step()

            # Accumulate metrics
            total_loss += batch_loss.item()
            total_acc += batch_acc.item()
            total_f1 += batch_f1

            # Logging
            if iter_idx % self.args.log_freq == 0 or iter_idx == num_batches - 1:
                write_log(
                    self.logger,
                    f"TRAIN - Epoch [{epoch_idx}/{self.args.num_epochs}] - "
                    f"Iter [{iter_idx}/{num_batches}] - Loss: {batch_loss.item():.4f}"
                )
                write_log(
                    self.logger,
                    f"TRAIN - Epoch [{epoch_idx}/{self.args.num_epochs}] - "
                    f"Iter [{iter_idx}/{num_batches}] - Acc: {batch_acc.item():.4f}"
                )
                write_log(
                    self.logger,
                    f"TRAIN - Epoch [{epoch_idx}/{self.args.num_epochs}] - "
                    f"Iter [{iter_idx}/{num_batches}] - F1: {batch_f1:.4f}"
                )

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches,
            'f1': total_f1 / num_batches
        }

    def validate(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        epoch_idx: int = 0
    ) -> Dict[str, float]:
        """
        Run validation.

        Args:
            dataloader: Validation data loader
            loss_fn: Loss function
            epoch_idx: Current epoch index (for logging)

        Returns:
            Dictionary of validation metrics (loss, accuracy, f1)
        """
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for iter_idx, data_dicts in enumerate(tqdm(
                dataloader,
                total=num_batches,
                desc=f'Validating - Epoch [{epoch_idx}/{self.args.num_epochs}]',
                position=0,
                leave=True
            )):
                # Move data to device
                input_ids = data_dicts['input_ids'].to(self.device)
                attention_mask = data_dicts['attention_mask'].to(self.device)
                token_type_ids = data_dicts['token_type_ids'].to(self.device)
                labels = data_dicts['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask, token_type_ids)

                # Compute loss and metrics
                batch_loss = loss_fn(logits, labels)
                batch_acc = (logits.argmax(dim=-1) == labels).float().mean()
                batch_f1 = f1_score(
                    labels.cpu().numpy(),
                    logits.argmax(dim=-1).cpu().numpy(),
                    average='macro'
                )

                # Accumulate metrics
                total_loss += batch_loss.item()
                total_acc += batch_acc.item()
                total_f1 += batch_f1

                # Logging
                if iter_idx % self.args.log_freq == 0 or iter_idx == num_batches - 1:
                    write_log(
                        self.logger,
                        f"VALID - Epoch [{epoch_idx}/{self.args.num_epochs}] - "
                        f"Iter [{iter_idx}/{num_batches}] - Loss: {batch_loss.item():.4f}"
                    )
                    write_log(
                        self.logger,
                        f"VALID - Epoch [{epoch_idx}/{self.args.num_epochs}] - "
                        f"Iter [{iter_idx}/{num_batches}] - Acc: {batch_acc.item():.4f}"
                    )
                    write_log(
                        self.logger,
                        f"VALID - Epoch [{epoch_idx}/{self.args.num_epochs}] - "
                        f"Iter [{iter_idx}/{num_batches}] - F1: {batch_f1:.4f}"
                    )

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches,
            'f1': total_f1 / num_batches
        }

    @staticmethod
    def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            logits: Model output logits
            labels: Ground truth labels

        Returns:
            Dictionary of metrics (accuracy, f1, precision, recall)
        """
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        return {
            'accuracy': (logits.argmax(dim=-1) == labels).float().mean().item(),
            'f1': f1_score(labels_np, preds, average='macro'),
            'precision': precision_score(labels_np, preds, average='macro', zero_division=0),
            'recall': recall_score(labels_np, preds, average='macro', zero_division=0)
        }

    def get_objective_value(self, metrics: Dict[str, float]) -> float:
        """
        Get the objective value to optimize based on args.optimize_objective.

        Args:
            metrics: Dictionary containing loss, accuracy, f1

        Returns:
            Objective value (higher is better)
        """
        if self.args.optimize_objective == 'loss':
            return -1 * metrics['loss']
        elif self.args.optimize_objective == 'accuracy':
            return metrics['accuracy']
        elif self.args.optimize_objective == 'f1':
            return metrics['f1']
        else:
            raise NotImplementedError(
                f"Unknown optimize_objective: {self.args.optimize_objective}"
            )
