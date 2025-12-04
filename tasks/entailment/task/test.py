"""
Entailment Task - Testing Module

Uses BaseEvaluator for common evaluation logic.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import argparse
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader

from tasks.entailment.model.model import EntailmentModel
from tasks.shared import CustomDataset, BaseEvaluator
from core import (
    TqdmLoggingHandler, write_log, get_torch_device, worker_init_fn,
    WandbManager, create_wandb_config, PiFiLogger,
    load_checkpoint,
)
from pifi.paths import get_model_path, get_preprocessed_dir


def testing(args: argparse.Namespace) -> tuple:
    device = get_torch_device(args.device)

    # Setup logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load dataset
    write_log(logger, "Loading dataset")
    preprocessed_dir = get_preprocessed_dir(args, dataset=args.test_dataset)
    dataset_test = CustomDataset(os.path.join(preprocessed_dir, 'test_processed.pkl'))
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )
    args.vocab_size = dataset_test.vocab_size
    args.num_classes = dataset_test.num_classes
    args.pad_token_id = dataset_test.pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

    # Build model
    write_log(logger, "Building model")
    model = EntailmentModel(args).to(device)

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_path = get_model_path(args)
    model = model.to('cpu')
    checkpoint = load_checkpoint(load_model_path)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_path}")
    del checkpoint

    # Initialize W&B
    wandb_mgr = None
    if args.use_wandb:
        wandb_mgr = WandbManager.get_instance()
        wandb_config = create_wandb_config(args, job_type='test')
        wandb_mgr.init(wandb_config)

    cls_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps)
    write_log(logger, f"Loss function: {cls_loss}")

    # Use BaseEvaluator for testing
    evaluator = BaseEvaluator(args, model, device, logger)
    metrics = evaluator.evaluate(dataloader_test, cls_loss)

    # Log test results using unified PiFiLogger
    if getattr(args, 'log_test_local', True) or args.use_wandb:
        try:
            pifi_logger = PiFiLogger.from_args(args)
            metadata = {
                'task': args.task,
                'dataset': args.task_dataset,
                'test_dataset': args.test_dataset,
                'model_type': args.model_type,
                'method': args.method,
                'llm_model': getattr(args, 'llm_model', ''),
                'layer_num': getattr(args, 'layer_num', -1),
                'seed': getattr(args, 'seed', 2023)
            }
            result_path = pifi_logger.log_test_results(metrics, metadata)
            write_log(logger, f"Test results saved to: {result_path}")
        except Exception as e:
            write_log(logger, f"Failed to save test results: {e}")

    # Finish W&B run if active
    if args.use_wandb and wandb_mgr:
        wandb_mgr.finish()

    return metrics['accuracy'], metrics['f1']
