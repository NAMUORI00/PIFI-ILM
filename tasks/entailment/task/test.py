import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging
import argparse
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.model import EntailmentModel
from model.dataset import CustomDataset
from utils.utils import TqdmLoggingHandler, write_log, get_torch_device, worker_init_fn

# Import from core modules
try:
    from core.wandb_manager import WandbManager, create_wandb_config
    from core.checkpoint import get_model_path, load_checkpoint
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from core.wandb_manager import WandbManager, create_wandb_config
    from core.checkpoint import get_model_path, load_checkpoint


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
    dataset_test = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, 'test_processed.pkl'))
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
    model_dir = get_model_path(args)
    load_model_name = os.path.join(model_dir, 'final_model.pt')
    model = model.to('cpu')
    checkpoint = load_checkpoint(load_model_name)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    # Initialize W&B
    wandb_mgr = None
    if args.use_wandb:
        wandb_mgr = WandbManager.get_instance()
        wandb_config = create_wandb_config(args, job_type='test')
        wandb_mgr.init(wandb_config)

    cls_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps)
    write_log(logger, f"Loss function: {cls_loss}")

    # Testing loop
    model = model.eval()
    test_loss_cls = 0
    test_acc_cls = 0
    test_f1_cls = 0
    test_precision_cls = 0
    test_recall_cls = 0

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc="Testing", position=0, leave=True)):
        input_ids = data_dicts['input_ids'].to(device)
        attention_mask = data_dicts['attention_mask'].to(device)
        token_type_ids = data_dicts['token_type_ids'].to(device)
        labels = data_dicts['labels'].to(device)

        with torch.no_grad():
            classification_logits = model(input_ids, attention_mask, token_type_ids)

        batch_loss_cls = cls_loss(classification_logits, labels)
        batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
        preds = classification_logits.argmax(dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        batch_f1_cls = f1_score(labels_np, preds, average='macro')
        batch_precision_cls = precision_score(labels_np, preds, average='macro', zero_division=0)
        batch_recall_cls = recall_score(labels_np, preds, average='macro', zero_division=0)

        test_loss_cls += batch_loss_cls.item()
        test_acc_cls += batch_acc_cls.item()
        test_f1_cls += batch_f1_cls
        test_precision_cls += batch_precision_cls
        test_recall_cls += batch_recall_cls

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_test) - 1:
            write_log(
                logger,
                f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - "
                f"Loss: {batch_loss_cls.item():.4f} | Acc: {batch_acc_cls.item():.4f} | "
                f"Prec: {batch_precision_cls:.4f} | Rec: {batch_recall_cls:.4f} | F1: {batch_f1_cls:.4f}"
            )

    # Compute final metrics
    test_loss_cls /= len(dataloader_test)
    test_acc_cls /= len(dataloader_test)
    test_f1_cls /= len(dataloader_test)
    test_precision_cls /= len(dataloader_test)
    test_recall_cls /= len(dataloader_test)

    write_log(
        logger,
        f"Done! - TEST - Loss: {test_loss_cls:.4f} - Acc: {test_acc_cls:.4f} - "
        f"Prec: {test_precision_cls:.4f} - Rec: {test_recall_cls:.4f} - F1: {test_f1_cls:.4f}"
    )

    # Log to W&B
    if args.use_wandb and wandb_mgr:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Model': [args.model_type],
            'Acc': [test_acc_cls],
            'Precision': [test_precision_cls],
            'Recall': [test_recall_cls],
            'F1': [test_f1_cls],
            'Loss': [test_loss_cls]
        })
        wandb_mgr.log_table('TEST_Result', wandb_df)
        wandb_mgr.log({
            'TEST/Loss': test_loss_cls,
            'TEST/Acc': test_acc_cls,
            'TEST/F1': test_f1_cls,
            'TEST/Precision': test_precision_cls,
            'TEST/Recall': test_recall_cls
        })
        wandb_mgr.finish()

    return test_acc_cls, test_f1_cls
