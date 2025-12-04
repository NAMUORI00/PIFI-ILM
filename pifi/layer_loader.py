"""
LLM Layer Loading Utilities for PiFi

This module provides functions to:
- Map model type strings to HuggingFace model IDs
- Load specific layers from LLM models for PiFi injection
"""

import argparse
from typing import Tuple

import torch.nn as nn
from transformers import AutoModelForCausalLM


# Model Registry: Maps short names to HuggingFace model IDs
MODEL_REGISTRY = {
    # SLMs (Small Language Models)
    'bert': 'google-bert/bert-base-uncased',
    'bert_large': 'google-bert/bert-large-uncased',
    'modern_bert': 'answerdotai/ModernBERT-base',
    'smollm': 'HuggingFaceTB/SmolLM2-135M',
    'roberta': 'FacebookAI/roberta-base',
    'roberta-large': 'FacebookAI/roberta-large',
    'electra': 'google/electra-base-discriminator',
    'albert': 'albert-base-v2',
    'deberta': 'microsoft/deberta-base',
    'debertav3': 'microsoft/deberta-v3-base',
    'mbert': 'google-bert/bert-base-multilingual-cased',
    'kcbert': 'beomi/kcbert-base',
    # Other encoder models
    'cnn': 'google-bert/bert-base-uncased',
    'lstm': 'google-bert/bert-base-uncased',
    'gru': 'google-bert/bert-base-uncased',
    'rnn': 'google-bert/bert-base-uncased',
    'transformer_enc': 'google-bert/bert-base-uncased',
    # Seq2Seq models
    'bart': 'facebook/bart-large',
    't5': 't5-large',
    # LLMs (Large Language Models)
    'llama2': 'meta-llama/Llama-2-7b-hf',
    'llama3': 'meta-llama/Meta-Llama-3-8B',
    'llama3.1': 'meta-llama/Meta-Llama-3.1-8B',
    'llama3.1_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'mistral0.1': 'mistralai/Mistral-7B-v0.1',
    'mistral0.3': 'mistralai/Mistral-7B-v0.3',
    'qwen2_7b': 'Qwen/Qwen2-7B',
    'qwen2_0.5b': 'Qwen/Qwen2-0.5B',
    'qwen2_1.5b': 'Qwen/Qwen2-1.5B',
    'qwen2_72b': 'Qwen/Qwen2-72B',
    'gemma2': 'google/gemma-2-9b',
    'falcon': 'tiiuae/falcon-7b',
    'kollama': 'beomi/Llama-3-Open-Ko-8B',
    'gerllama': 'DiscoResearch/Llama3-German-8B',
    'chillama': 'hfl/llama-3-chinese-8b',
}


def get_huggingface_model_name(model_type: str) -> str:
    """
    Get HuggingFace model ID from model type string.

    Args:
        model_type: Short model name (e.g., 'bert', 'llama2')

    Returns:
        HuggingFace model ID (e.g., 'google-bert/bert-base-uncased')

    Raises:
        NotImplementedError: If model type is not supported
    """
    name = model_type.lower()
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported")


def llm_layer(llm_model_name: str, args: argparse.Namespace) -> Tuple[nn.Module, int, int]:
    """
    Load and return a specific layer from an LLM model.

    This is the core function for PiFi's layer injection mechanism.
    It loads a pre-trained LLM and extracts a specific transformer layer
    for injection into the SLM architecture.

    Args:
        llm_model_name: HuggingFace model ID of the LLM
        args: Arguments containing:
            - cache_path: Path to cache downloaded models
            - freeze: Whether to freeze LLM parameters
            - llm_model: Name of LLM model (for architecture detection)
            - layer_num: Index of the layer to extract

    Returns:
        Tuple of (llm_layer, llm_embed_size, llm_hidden_size)
        - llm_layer: The extracted transformer layer module
        - llm_embed_size: Embedding dimension of the LLM
        - llm_hidden_size: Hidden dimension of the LLM

    Raises:
        ValueError: If layer_num exceeds the number of layers in the model
    """
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        cache_dir=args.cache_path
    )

    # Freeze LLM parameters if specified
    if args.freeze:
        for param in llm_model.parameters():
            param.requires_grad = False

    # Get the specific layer based on model architecture
    if args.llm_model == 'falcon':
        # Falcon uses transformer.h for layers
        if len(llm_model.transformer.h) <= args.layer_num:
            raise ValueError(
                f"Layer {args.layer_num} does not exist in the model. "
                f"Model has {len(llm_model.transformer.h)} layers."
            )
        layer = llm_model.transformer.h[args.layer_num]
    else:
        # Most LLMs (LLaMA, Mistral, Qwen, etc.) use model.layers
        if len(llm_model.model.layers) <= args.layer_num:
            raise ValueError(
                f"Layer {args.layer_num} does not exist in the model. "
                f"Model has {len(llm_model.model.layers)} layers."
            )
        layer = llm_model.model.layers[args.layer_num]

    llm_embed_size = llm_model.config.hidden_size
    llm_hidden_size = llm_model.config.hidden_size

    return layer, llm_embed_size, llm_hidden_size
