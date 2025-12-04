"""
Base Evaluator for PiFi Tasks

Provides common evaluation/testing logic that can be used by
classification and entailment tasks.
"""

import logging
from typing import Dict, Optional
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from core import write_log


class BaseEvaluator:
    """
    Base evaluator providing common testing/evaluation logic.

    This class extracts the shared evaluation loop logic from task-specific
    test.py files. It handles:
    - Running evaluation on test set
    - Metrics computation (accuracy, F1, precision, recall, loss)
    - Logging during evaluation

    Usage:
        evaluator = BaseEvaluator(args, model, device, logger)
        metrics = evaluator.evaluate(test_loader, loss_fn)
    """

    def __init__(
        self,
        args,
        model: nn.Module,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the evaluator.

        Args:
            args: Parsed command line arguments
            model: The model to evaluate
            device: Device to run evaluation on
            logger: Optional logger for output
        """
        self.args = args
        self.model = model
        self.device = device
        self.logger = logger

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        Run evaluation on the test set.

        Args:
            dataloader: Test data loader
            loss_fn: Loss function

        Returns:
            Dictionary of test metrics (loss, accuracy, f1, precision, recall)
        """
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        total_precision = 0
        total_recall = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for iter_idx, data_dicts in enumerate(tqdm(
                dataloader,
                total=num_batches,
                desc="Testing",
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

                # Compute loss
                batch_loss = loss_fn(logits, labels)

                # Compute metrics
                batch_acc = (logits.argmax(dim=-1) == labels).float().mean()
                preds = logits.argmax(dim=-1).cpu().numpy()
                labels_np = labels.cpu().numpy()

                batch_f1 = f1_score(labels_np, preds, average='macro')
                batch_precision = precision_score(labels_np, preds, average='macro', zero_division=0)
                batch_recall = recall_score(labels_np, preds, average='macro', zero_division=0)

                # Accumulate metrics
                total_loss += batch_loss.item()
                total_acc += batch_acc.item()
                total_f1 += batch_f1
                total_precision += batch_precision
                total_recall += batch_recall

                # Logging
                if iter_idx % self.args.log_freq == 0 or iter_idx == num_batches - 1:
                    write_log(
                        self.logger,
                        f"TEST - Iter [{iter_idx}/{num_batches}] - "
                        f"Loss: {batch_loss.item():.4f} | Acc: {batch_acc.item():.4f} | "
                        f"Prec: {batch_precision:.4f} | Rec: {batch_recall:.4f} | F1: {batch_f1:.4f}"
                    )

        # Compute averages
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches,
            'f1': total_f1 / num_batches,
            'precision': total_precision / num_batches,
            'recall': total_recall / num_batches
        }

        # Log final results
        write_log(
            self.logger,
            f"Done! - TEST - Loss: {metrics['loss']:.4f} - Acc: {metrics['accuracy']:.4f} - "
            f"Prec: {metrics['precision']:.4f} - Rec: {metrics['recall']:.4f} - F1: {metrics['f1']:.4f}"
        )

        return metrics

    @staticmethod
    def compute_batch_metrics(
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        Compute metrics for a single batch.

        Args:
            logits: Model output logits
            labels: Ground truth labels
            loss_fn: Loss function

        Returns:
            Dictionary of batch metrics
        """
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        return {
            'loss': loss_fn(logits, labels).item(),
            'accuracy': (logits.argmax(dim=-1) == labels).float().mean().item(),
            'f1': f1_score(labels_np, preds, average='macro'),
            'precision': precision_score(labels_np, preds, average='macro', zero_division=0),
            'recall': recall_score(labels_np, preds, average='macro', zero_division=0)
        }
