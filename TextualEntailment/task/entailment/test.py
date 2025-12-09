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
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.entailment.model import EntailmentModel
from model.entailment.dataset import CustomDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device

def testing(args: argparse.Namespace) -> tuple:
    device = get_torch_device(args.device)

    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    write_log(logger, "Loading dataset")
    dataset_test = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'test_processed.pkl'))
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_test.vocab_size
    args.num_classes = dataset_test.num_classes
    args.pad_token_id = dataset_test.pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

    write_log(logger, "Building model")
    model = EntailmentModel(args).to(device)

    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.padding, args.model_type, args.method, args.llm_model, str(args.layer_num), 'final_model.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args) + f' - Test',
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Model: {args.model_type}"])

    cls_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps)
    write_log(logger, f"Loss function: {cls_loss}")

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
        labels = data_dicts['labels'].to(device) # For calculating accuracy

        with torch.no_grad():
            classification_logits = model(input_ids, attention_mask, token_type_ids)

        preds = classification_logits.argmax(dim=-1)

        batch_loss_cls = cls_loss(classification_logits, labels)
        batch_acc_cls = (preds == labels).float().mean()
        batch_f1_cls = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        batch_precision_cls = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
        batch_recall_cls = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

        test_loss_cls += batch_loss_cls.item()
        test_acc_cls += batch_acc_cls.item()
        test_f1_cls += batch_f1_cls
        test_precision_cls += batch_precision_cls
        test_recall_cls += batch_recall_cls

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Loss: {batch_loss_cls.item():.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Acc: {batch_acc_cls.item():.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - F1: {batch_f1_cls:.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Precision: {batch_precision_cls:.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Recall: {batch_recall_cls:.4f}")
    print(len(dataloader_test))
    test_loss_cls /= len(dataloader_test)
    test_acc_cls /= len(dataloader_test)
    test_f1_cls /= len(dataloader_test)
    test_precision_cls /= len(dataloader_test)
    test_recall_cls /= len(dataloader_test)

    write_log(logger, f"Done! - TEST - Loss: {test_loss_cls:.4f} - Acc: {test_acc_cls:.4f} - F1: {test_f1_cls:.4f} - Precision: {test_precision_cls:.4f} - Recall: {test_recall_cls:.4f}")
    if args.use_tensorboard:
        writer.add_scalar('TEST/Loss', test_loss_cls, 0)
        writer.add_scalar('TEST/Acc', test_acc_cls, 0)
        writer.add_scalar('TEST/F1', test_f1_cls, 0)
        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Model': [args.model_type],
            'Acc': [test_acc_cls],
            'F1': [test_f1_cls],
            'Loss': [test_loss_cls]
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({'TEST_Result': wandb_table})

        wandb.finish()
    # Save result to txt
    result_line = (
        f"dataset={args.task_dataset}  "
        f"model={args.model_type}  "
        f"llm={args.llm_model}  "
        f"layer={args.layer_num}  "
        f"acc={test_acc_cls:.4f}  "
        f"precision={test_precision_cls:.4f}  "
        f"recall={test_recall_cls:.4f}  "
        f"f1={test_f1_cls:.4f}  "
        f"loss={test_loss_cls:.4f}\n"
    )

    os.makedirs(args.result_path, exist_ok=True)
    result_txt_path = os.path.join(
        args.result_path,
        f"{args.task_dataset}_{args.model_type}_{args.llm_model}_results.txt"
    )
    with open(result_txt_path, "a", encoding="utf-8") as f:
        f.write(result_line)

    write_log(logger, f"Result saved to {result_txt_path}")

    return test_acc_cls, test_f1_cls
