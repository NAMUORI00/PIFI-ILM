"""
Entailment Model for PiFi

Inherits from BasePiFiModel and provides entailment-specific functionality.
"""

import argparse
import torch

from pifi.model import BasePiFiModel


class EntailmentModel(BasePiFiModel):
    """
    Entailment (NLI) task model.

    Inherits all functionality from BasePiFiModel.
    The base class handles:
    - SLM loading and configuration
    - LLM layer loading for PiFi method
    - Pooling strategies
    - Classification head
    - Forward pass logic

    This class exists to maintain the task-specific interface and
    allow for any entailment-specific customizations if needed.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the entailment model.

        Args:
            args: Parsed command line arguments containing:
                - model_type: Type of SLM to use
                - model_ispretrained: Whether to use pretrained weights
                - cache_path: Path to cache models
                - method: 'base' or 'pifi'
                - llm_model: LLM model type (for pifi method)
                - layer_num: LLM layer index (for pifi method)
                - num_classes: Number of entailment classes (typically 3)
                - dropout_rate: Dropout rate for classifier
                - padding: Pooling strategy
        """
        super(EntailmentModel, self).__init__(args)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for entailment classification.

        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            token_type_ids: Token type IDs (batch_size, seq_length)

        Returns:
            Entailment classification logits (batch_size, num_classes)
        """
        # Use the base class forward method
        return super().forward(input_ids, attention_mask, token_type_ids)
