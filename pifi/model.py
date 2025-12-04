"""
Base PiFi Model

This module provides the BasePiFiModel class that contains shared logic
for all PiFi task models. Classification and Entailment models inherit
from this base class.
"""

import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .layer_loader import get_huggingface_model_name, llm_layer


class BasePiFiModel(nn.Module):
    """
    Base model for PiFi (Plugin Fine-tuning) methodology.

    This class provides:
    - SLM (Small Language Model) loading and configuration
    - LLM layer loading and injection for PiFi method
    - Pooling strategies (CLS, average with/without padding)
    - Dimension mapping for LLM layer injection

    Subclasses (ClassificationModel, EntailmentModel) inherit this and
    implement task-specific forward methods if needed.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super(BasePiFiModel, self).__init__()
        self.args = args

        # Load SLM (Small Language Model)
        huggingface_model_name = get_huggingface_model_name(self.args.model_type)
        self.config = AutoConfig.from_pretrained(
            huggingface_model_name,
            cache_dir=self.args.cache_path
        )

        if args.model_ispretrained:
            self.model = AutoModel.from_pretrained(
                huggingface_model_name,
                cache_dir=self.args.cache_path
            )
        else:
            self.model = AutoModel.from_config(self.config)

        self.embed_size = self.model.config.hidden_size
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = self.args.num_classes

        # Load LLM layer for PiFi method
        if self.args.method == 'pifi':
            llm_model_name = get_huggingface_model_name(self.args.llm_model)
            self.llm_layer, self.llm_embed_size, self.llm_hidden_size = llm_layer(
                llm_model_name, args
            )
            # Dimension mappers for LLM injection
            self.llama_dim_mapper1 = nn.Linear(self.embed_size, self.llm_embed_size, bias=False)
            self.llama_dim_mapper2 = nn.Linear(self.llm_embed_size, self.embed_size, bias=False)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.Dropout(self.args.dropout_rate),
            nn.GELU(),
            nn.Linear(self.embed_size, self.num_classes),
        )

    def get_pooled_output(
        self,
        model_output,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply pooling strategy to get fixed-size representation.

        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask tensor

        Returns:
            Pooled output tensor of shape (batch_size, hidden_size)
        """
        if self.args.padding == 'cls':
            # Use [CLS] token representation
            return model_output.last_hidden_state[:, 0, :]

        elif self.args.padding == 'average_pooling_with_padding':
            # Simple mean over all tokens (including padding)
            return torch.mean(model_output.last_hidden_state, dim=1)

        elif self.args.padding == 'average_pooling_without_padding':
            # Mean over non-padding tokens only
            sum_output = torch.sum(
                model_output.last_hidden_state * attention_mask.unsqueeze(-1),
                dim=1
            )
            count_non_padding = torch.sum(attention_mask, dim=1, keepdim=True)
            return sum_output / count_non_padding

        else:
            # Default to CLS
            return model_output.last_hidden_state[:, 0, :]

    def apply_llm_layer(
        self,
        cls_output: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Apply LLM layer injection for PiFi method.

        Args:
            cls_output: Pooled output from SLM (batch_size, hidden_size)
            device: Device to run computations on

        Returns:
            LLM-enhanced output tensor (batch_size, hidden_size)
        """
        # Reshape for LLM layer: (batch, 1, hidden)
        cls_output = cls_output.unsqueeze(1).to(device)
        batch_size = cls_output.size(0)
        seq_length = cls_output.size(1)

        # Create attention mask and position ids
        attention_mask = torch.ones(
            (batch_size, seq_length),
            dtype=cls_output.dtype
        ).to(device)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1).to(device)

        # Map to LLM dimension
        cls_output = self.llama_dim_mapper1(cls_output)

        # Apply LLM layer
        if self.args.llm_model == 'falcon':
            llm_outputs = self.llm_layer(
                hidden_states=cls_output,
                attention_mask=attention_mask[:, None, None, :],
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
                alibi=False
            )
        else:
            llm_outputs = self.llm_layer(
                hidden_states=cls_output,
                attention_mask=attention_mask[:, None, None, :],
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )

        # Extract output and map back to SLM dimension
        llm_outputs = llm_outputs[0].squeeze(1)
        llm_outputs = self.llama_dim_mapper2(llm_outputs)

        return llm_outputs

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input and apply pooling.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional, for BERT-like models)

        Returns:
            Pooled representation tensor
        """
        # Some models don't use token_type_ids
        if self.args.model_type in ['modern_bert', 'smollm']:
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        else:
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )

        return self.get_pooled_output(model_output, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            token_type_ids: Token type IDs (batch_size, seq_length)

        Returns:
            Classification logits (batch_size, num_classes)
        """
        device = input_ids.device

        # Encode input
        cls_output = self.encode(input_ids, attention_mask, token_type_ids)

        # Apply method-specific processing
        if self.args.method == 'base':
            classification_logits = self.classifier(cls_output)
        elif self.args.method == 'pifi':
            llm_output = self.apply_llm_layer(cls_output, device)
            classification_logits = self.classifier(llm_output)
        else:
            # Default to base method
            classification_logits = self.classifier(cls_output)

        return classification_logits
