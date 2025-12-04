import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import shutil
import logging
import argparse
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.entailment.model import EntailmentModel
from model.entailment.dataset import CustomDataset
from model.optimizer.optimizer import get_optimizer
from model.optimizer.scheduler import get_scheduler
from utils.utils import TqdmLoggingHandler, write_log, get_wandb_exp_name, get_torch_device, check_path, worker_init_fn

# Import from core modules
try:
    from core.wandb_manager import WandbManager, create_wandb_config
    from core.checkpoint import (
        save_checkpoint, load_checkpoint, restore_rng_states,
        create_metadata_from_args, get_checkpoint_path, get_model_path
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from core.wandb_manager import WandbManager, create_wandb_config
    from core.checkpoint import (
        save_checkpoint, load_checkpoint, restore_rng_states,
        create_metadata_from_args, get_checkpoint_path, get_model_path
    )

try:
    from selection import auto_select_layer
except Exception:
    auto_select_layer = None


def training(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)
    print(f"Device: {device}")

    # Setup logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load data
    write_log(logger, "Loading data")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['train'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, 'train_processed.pkl'))
    dataset_dict['valid'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, 'valid_processed.pkl'))

    # Create reproducible DataLoaders
    train_generator = torch.Generator().manual_seed(args.seed)
    dataloader_dict['train'] = DataLoader(
        dataset_dict['train'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=train_generator
    )
    dataloader_dict['valid'] = DataLoader(
        dataset_dict['valid'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

    args.vocab_size = dataset_dict['train'].vocab_size
    args.num_classes = dataset_dict['train'].num_classes
    args.pad_token_id = dataset_dict['train'].pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_dict['train'])} / {len(dataloader_dict['train'])}")
    write_log(logger, f"Valid dataset size / iterations: {len(dataset_dict['valid'])} / {len(dataloader_dict['valid'])}")

    # Optional: auto-select LLM layer before building model
    if getattr(args, 'auto_select_layer', False) and args.method == 'pifi' and getattr(args, 'layer_num', -1) < 0:
        if auto_select_layer is None:
            print("[selection] Module not available; skipping auto selection")
        else:
            try:
                sel_idx = auto_select_layer(args)
                args.layer_num = int(sel_idx)
                print(f"[selection] Using auto-selected LLM layer: {args.layer_num}")
            except Exception as e:
                print(f"[selection] Failed to auto-select layer: {e}")

    # Build model
    write_log(logger, "Building model")
    model = EntailmentModel(args).to(device)

    # Build optimizer and scheduler
    write_log(logger, "Building optimizer and scheduler")
    optimizer = get_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay, optim_type=args.optimizer)
    scheduler = get_scheduler(
        optimizer, len(dataloader_dict['train']),
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler
    )
    write_log(logger, f"Optimizer: {optimizer}")
    write_log(logger, f"Scheduler: {scheduler}")

    cls_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps)
    write_log(logger, f"Loss function: {cls_loss}")
    write_log(logger, f"Method: {args.method}")

    # Initialize W&B manager
    wandb_mgr = None
    if args.use_wandb:
        wandb_mgr = WandbManager.get_instance()

    # Resume training if needed
    start_epoch = 0
    checkpoint_dir = get_checkpoint_path(args)
    resume_wandb_id = None
    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0

    if args.job == 'resume_training':
        write_log(logger, "Resuming training model")
        # Prefer last.pt (latest state) over checkpoint.pt (best state)
        last_checkpoint_file = os.path.join(checkpoint_dir, 'last.pt')
        best_checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if os.path.exists(last_checkpoint_file):
            checkpoint_file = last_checkpoint_file
            write_log(logger, "Found last.pt, resuming from latest state")
        elif os.path.exists(best_checkpoint_file):
            checkpoint_file = best_checkpoint_file
            write_log(logger, "last.pt not found, falling back to checkpoint.pt (best state)")
        else:
            checkpoint_file = None

        if checkpoint_file is None:
            write_log(logger, f"No checkpoint found in {checkpoint_dir}, starting from scratch")
        else:
            model = model.to('cpu')
            checkpoint = load_checkpoint(checkpoint_file)
            metadata = checkpoint['metadata']
            start_epoch = metadata.epoch + 1
            best_epoch_idx = getattr(metadata, 'best_epoch_idx', metadata.epoch)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None and checkpoint['scheduler'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])

            # Restore RNG states for reproducibility
            if metadata.torch_rng_state is not None:
                restore_rng_states(metadata)

            # Restore best metric and early stopping state
            metric_name = getattr(metadata, 'best_metric_name', None) or args.optimize_objective
            checkpoint_best_metric = getattr(metadata, 'best_metric', None)
            checkpoint_best_objective = None
            if checkpoint_best_metric is None:
                # Fallback for older checkpoints that may store the signed objective directly
                checkpoint_best_objective = checkpoint.get('best_valid_objective_value')

            if checkpoint_best_metric is not None:
                best_valid_objective_value = -checkpoint_best_metric if metric_name == 'loss' else checkpoint_best_metric
                write_log(
                    logger,
                    f"Restored best {metric_name} from checkpoint: {abs(best_valid_objective_value):.4f} at epoch {best_epoch_idx}"
                )
            elif checkpoint_best_objective is not None:
                best_valid_objective_value = checkpoint_best_objective
                write_log(
                    logger,
                    f"Restored best objective from checkpoint: {abs(best_valid_objective_value):.4f} at epoch {best_epoch_idx}"
                )
            else:
                write_log(logger, "Checkpoint missing best metric; continuing without restoration")

            restored_counter = getattr(metadata, 'early_stopping_counter', None)
            if restored_counter is None:
                restored_counter = checkpoint.get('early_stopping_counter', early_stopping_counter)
            early_stopping_counter = restored_counter
            write_log(logger, f"Restored early stopping counter: {early_stopping_counter}")

            resume_wandb_id = metadata.wandb_id
            model = model.to(device)
            write_log(logger, f"Loaded checkpoint from {checkpoint_file}")
            write_log(logger, f"Resuming from epoch {start_epoch}")
            del checkpoint

    # Initialize W&B
    if args.use_wandb:
        wandb_config = create_wandb_config(args, job_type='train')
        if resume_wandb_id:
            wandb_config.resume = True
            wandb_config.run_id = resume_wandb_id
        wandb_mgr.init(wandb_config, model=model, criterion=cls_loss)

    # Training loop
    write_log(logger, f"Start training from epoch {start_epoch}")
    for epoch_idx in range(start_epoch, args.num_epochs):
        model = model.train()
        train_loss_cls = 0
        train_acc_cls = 0
        train_f1_cls = 0

        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['train'], total=len(dataloader_dict['train']), desc=f'Training - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            input_ids = data_dicts['input_ids'].to(device)
            attention_mask = data_dicts['attention_mask'].to(device)
            token_type_ids = data_dicts['token_type_ids'].to(device)
            labels = data_dicts['labels'].to(device)

            classification_logits = model(input_ids, attention_mask, token_type_ids)

            batch_loss_cls = cls_loss(classification_logits, labels)
            batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
            batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

            optimizer.zero_grad()
            batch_loss_cls.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if args.scheduler in ['StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step()

            train_loss_cls += batch_loss_cls.item()
            train_acc_cls += batch_acc_cls.item()
            train_f1_cls += batch_f1_cls

            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['train']) - 1:
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Loss: {batch_loss_cls.item():.4f}")
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Acc: {batch_acc_cls.item():.4f}")
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - F1: {batch_f1_cls:.4f}")

        # Validation
        model = model.eval()
        valid_loss_cls = 0
        valid_acc_cls = 0
        valid_f1_cls = 0

        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc=f'Validating - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            input_ids = data_dicts['input_ids'].to(device)
            attention_mask = data_dicts['attention_mask'].to(device)
            token_type_ids = data_dicts['token_type_ids'].to(device)
            labels = data_dicts['labels'].to(device)

            with torch.no_grad():
                classification_logits = model(input_ids, attention_mask, token_type_ids)

            batch_loss_cls = cls_loss(classification_logits, labels)
            batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
            batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

            valid_loss_cls += batch_loss_cls.item()
            valid_acc_cls += batch_acc_cls.item()
            valid_f1_cls += batch_f1_cls

            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['valid']) - 1:
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Loss: {batch_loss_cls.item():.4f}")
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Acc: {batch_acc_cls.item():.4f}")
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - F1: {batch_f1_cls:.4f}")

        # Scheduler step
        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_loss_cls)

        # Compute epoch metrics
        valid_loss_cls /= len(dataloader_dict['valid'])
        valid_acc_cls /= len(dataloader_dict['valid'])
        valid_f1_cls /= len(dataloader_dict['valid'])

        # Determine objective value
        if args.optimize_objective == 'loss':
            valid_objective_value = -1 * valid_loss_cls
        elif args.optimize_objective == 'accuracy':
            valid_objective_value = valid_acc_cls
        elif args.optimize_objective == 'f1':
            valid_objective_value = valid_f1_cls
        else:
            raise NotImplementedError

        # Save best checkpoint
        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            write_log(logger, f"VALID - Saving checkpoint for best valid {args.optimize_objective}...")
            early_stopping_counter = 0

            check_path(checkpoint_dir)

            # Create metadata with wandb_id for resume support
            metadata = create_metadata_from_args(
                args,
                epoch=epoch_idx,
                wandb_id=wandb_mgr.run_id if wandb_mgr else None,
                best_metric=abs(best_valid_objective_value),
                best_metric_name=args.optimize_objective,
                best_epoch_idx=best_epoch_idx,
                early_stopping_counter=early_stopping_counter
            )

            save_checkpoint(
                os.path.join(checkpoint_dir, 'checkpoint.pt'),
                model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict() if scheduler is not None else None,
                metadata
            )

            write_log(logger, f"VALID - Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
            write_log(logger, f"VALID - Saved checkpoint to {checkpoint_dir}")
        else:
            early_stopping_counter += 1
            write_log(logger, f"VALID - Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")

        # Save last checkpoint (every epoch for reliable resume)
        check_path(checkpoint_dir)
        last_metadata = create_metadata_from_args(
            args,
            epoch=epoch_idx,
            wandb_id=wandb_mgr.run_id if wandb_mgr else None,
            best_metric=abs(best_valid_objective_value) if best_valid_objective_value is not None else None,
            best_metric_name=args.optimize_objective,
            best_epoch_idx=best_epoch_idx,
            early_stopping_counter=early_stopping_counter
        )
        save_checkpoint(
            os.path.join(checkpoint_dir, 'last.pt'),
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict() if scheduler is not None else None,
            last_metadata
        )

        # Log to W&B
        if args.use_wandb and wandb_mgr:
            wandb_mgr.log({
                'TRAIN/Epoch_Loss': train_loss_cls / len(dataloader_dict['train']),
                'TRAIN/Epoch_Acc': train_acc_cls / len(dataloader_dict['train']),
                'TRAIN/Epoch_F1': train_f1_cls / len(dataloader_dict['train']),
                'VALID/Epoch_Loss': valid_loss_cls,
                'VALID/Epoch_Acc': valid_acc_cls,
                'VALID/Epoch_F1': valid_f1_cls,
                'Learning_Rate': optimizer.param_groups[0]['lr'],
                'Epoch_Index': epoch_idx
            })
            wandb_mgr.alert(
                title='Epoch End',
                text=f"VALID - Epoch {epoch_idx} - Loss: {valid_loss_cls:.4f} - Acc: {valid_acc_cls:.4f}",
                level='INFO',
                wait_duration=300
            )

        # Early stopping check
        if early_stopping_counter >= args.early_stopping_patience:
            write_log(logger, f"VALID - Early stopping at epoch {epoch_idx}...")
            break

    write_log(logger, f"Done! Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")

    # Save final model
    final_model_dir = get_model_path(args)
    check_path(final_model_dir)
    shutil.copyfile(os.path.join(checkpoint_dir, 'checkpoint.pt'), os.path.join(final_model_dir, 'final_model.pt'))
    write_log(logger, f"FINAL - Saved final model to {final_model_dir}")

    # Finish W&B run
    if args.use_wandb and wandb_mgr:
        wandb_mgr.finish()
