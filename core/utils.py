"""
Unified Utility Functions for PiFi
Common utilities used across all tasks
"""

import os
import sys
import time
import tqdm
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def check_path(path: str):
    """
    Check if the path exists and create it if not.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_random_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic algorithms (may impact performance)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass  # Some operations don't support deterministic mode

        # cuBLAS deterministic mode (environment variable)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def worker_init_fn(worker_id: int):
    """
    Worker initialization function for DataLoader reproducibility.
    Each worker gets a unique but deterministic seed.

    Args:
        worker_id: DataLoader worker ID
    """
    # Get base seed from PyTorch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_torch_device(device: str):
    """
    Get PyTorch device (CUDA, MPS, or CPU)
    """
    if device is not None:
        get_torch_device.device = device

    if 'cuda' in get_torch_device.device:
        if torch.cuda.is_available():
            return torch.device(get_torch_device.device)
        else:
            print("No GPU found. Using CPU.")
            return torch.device('cpu')
    elif 'mps' in device:
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install"
                      " was not built with MPS enabled.")
                print("Using CPU.")
            else:
                print("MPS not available because the current MacOS version"
                      " is not 12.3+ and/or you do not have an MPS-enabled"
                      " device on this machine.")
                print("Using CPU.")
            return torch.device('cpu')
        else:
            return torch.device(get_torch_device.device)
    elif 'cpu' in device:
        return torch.device('cpu')
    else:
        print("No such device found. Using CPU.")
        return torch.device('cpu')


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works well with tqdm progress bars"""
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)


def write_log(logger, message):
    """Write message to logger if logger exists"""
    if logger:
        logger.info(message)


def get_wandb_exp_name(args: argparse.Namespace):
    """
    Get the experiment name for weights and biases experiment.
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


def get_huggingface_model_name(model_type: str) -> str:
    """
    Get HuggingFace model ID from model type string
    """
    name = model_type.lower()

    # SLMs (Small Language Models)
    if name in ['bert', 'cnn', 'lstm', 'gru', 'rnn', 'transformer_enc']:
        return 'google-bert/bert-base-uncased'
    elif name == 'bert_large':
        return 'google-bert/bert-large-uncased'
    elif name == 'modern_bert':
        return 'answerdotai/ModernBERT-base'
    elif name == 'smollm':
        return 'HuggingFaceTB/SmolLM2-135M'
    elif name == 'roberta':
        return 'FacebookAI/roberta-base'
    elif name == 'roberta-large':
        return 'FacebookAI/roberta-large'
    elif name == 'electra':
        return 'google/electra-base-discriminator'
    elif name == 'albert':
        return 'albert-base-v2'
    elif name == 'deberta':
        return 'microsoft/deberta-base'
    elif name == 'debertav3':
        return 'microsoft/deberta-v3-base'
    elif name == 'mbert':
        return 'google-bert/bert-base-multilingual-cased'
    elif name == 'kcbert':
        return 'beomi/kcbert-base'

    # Other models
    elif name == 'bart':
        return 'facebook/bart-large'
    elif name == 't5':
        return 't5-large'

    # LLMs (Large Language Models)
    elif name == 'llama2':
        return 'meta-llama/Llama-2-7b-hf'
    elif name == 'llama3':
        return 'meta-llama/Meta-Llama-3-8B'
    elif name == 'llama3.1':
        return 'meta-llama/Meta-Llama-3.1-8B'
    elif name == 'llama3.1_instruct':
        return 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif name == 'mistral0.1':
        return 'mistralai/Mistral-7B-v0.1'
    elif name == 'mistral0.3':
        return 'mistralai/Mistral-7B-v0.3'
    elif name == 'qwen2_7b':
        return 'Qwen/Qwen2-7B'
    elif name == 'qwen2_0.5b':
        return 'Qwen/Qwen2-0.5B'
    elif name == 'qwen2_1.5b':
        return 'Qwen/Qwen2-1.5B'
    elif name == 'qwen2_72b':
        return 'Qwen/Qwen2-72B'
    elif name == 'gemma2':
        return 'google/gemma-2-9b'
    elif name == 'falcon':
        return 'tiiuae/falcon-7b'
    elif name == 'kollama':
        return 'beomi/Llama-3-Open-Ko-8B'
    elif name == 'gerllama':
        return 'DiscoResearch/Llama3-German-8B'
    elif name == 'chillama':
        return 'hfl/llama-3-chinese-8b'
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported")


def llm_layer(llm_model_name, args):
    """
    Load and return a specific layer from an LLM model.
    Returns: (llm_layer, llm_embed_size, llm_hidden_size)
    """
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=args.cache_path)

    # Freeze LLM parameters if specified
    if args.freeze:
        for param in llm_model.parameters():
            param.requires_grad = False

    # Get the specific layer
    if args.llm_model == 'falcon':
        if len(llm_model.transformer.h) <= args.layer_num:
            raise ValueError(f"Layer {args.layer_num} does not exist in the model. Training halted.")
        else:
            llm_layer = llm_model.transformer.h[args.layer_num]
    else:
        if len(llm_model.model.layers) <= args.layer_num:
            raise ValueError(f"Layer {args.layer_num} does not exist in the model. Training halted.")
        else:
            llm_layer = llm_model.model.layers[args.layer_num]

    llm_embed_size = llm_model.config.hidden_size
    llm_hidden_size = llm_model.config.hidden_size

    return llm_layer, llm_embed_size, llm_hidden_size
