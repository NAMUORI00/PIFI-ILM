"""
Classification Task - Training Module

Uses BaseTrainer for common training/validation logic.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
import logging
import argparse
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import DataLoader

from tasks.classification.model.model import ClassificationModel
from tasks.shared import CustomDataset, BaseTrainer
from core import (
    get_optimizer, get_scheduler,
    TqdmLoggingHandler, write_log, get_torch_device, check_path, worker_init_fn,
    WandbManager, create_wandb_config, PiFiLogger,
    save_checkpoint, load_checkpoint, restore_rng_states,
)
from core.checkpoint import create_metadata_from_args
from pifi.paths import get_checkpoint_dir, get_model_path, get_preprocessed_dir

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
    preprocessed_dir = get_preprocessed_dir(args)
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['train'] = CustomDataset(os.path.join(preprocessed_dir, 'train_processed.pkl'))
    dataset_dict['valid'] = CustomDataset(os.path.join(preprocessed_dir, 'valid_processed.pkl'))

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
    model = ClassificationModel(args).to(device)

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

    # Auto-resume: check for existing checkpoints
    start_epoch = 0
    checkpoint_dir = get_checkpoint_dir(args)
    resume_wandb_id = None
    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0

    # Auto-detect checkpoint and resume (works for both 'training' and 'resume_training')
    last_checkpoint_file = os.path.join(checkpoint_dir, 'last.pt')
    best_checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
    checkpoint_file = None

    if os.path.exists(last_checkpoint_file):
        checkpoint_file = last_checkpoint_file
        write_log(logger, "Found last.pt, will resume from latest state")
    elif os.path.exists(best_checkpoint_file):
        checkpoint_file = best_checkpoint_file
        write_log(logger, "Found checkpoint.pt, will resume from best state")

    if checkpoint_file is not None:
        model = model.to('cpu')
        checkpoint = load_checkpoint(checkpoint_file)
        metadata = checkpoint['metadata']
        start_epoch = metadata.epoch + 1
        best_epoch_idx = getattr(metadata, 'best_epoch_idx', metadata.epoch)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])

        if metadata.torch_rng_state is not None:
            restore_rng_states(metadata)

        if metadata.best_metric is not None:
            metric_name = metadata.best_metric_name or args.optimize_objective
            if metric_name == 'loss':
                best_valid_objective_value = -metadata.best_metric
            else:
                best_valid_objective_value = metadata.best_metric
            write_log(
                logger,
                f"Restored best {metric_name} from checkpoint: {abs(best_valid_objective_value):.4f} at epoch {best_epoch_idx}"
            )
        early_stopping_counter = getattr(metadata, 'early_stopping_counter', early_stopping_counter)

        resume_wandb_id = metadata.wandb_id
        model = model.to(device)
        write_log(logger, f"Loaded checkpoint from {checkpoint_file}")
        write_log(logger, f"Resuming from epoch {start_epoch}")
        del checkpoint

    # Skip if already completed
    if start_epoch >= args.num_epochs:
        write_log(logger, f"Training already completed ({start_epoch} >= {args.num_epochs} epochs). Skipping.")
        return

    # Initialize W&B
    if args.use_wandb:
        wandb_config = create_wandb_config(args, job_type='train')
        if resume_wandb_id:
            wandb_config.resume = True
            wandb_config.run_id = resume_wandb_id
        wandb_mgr.init(wandb_config, model=model, criterion=cls_loss)

    # Create trainer for training/validation loops
    trainer = BaseTrainer(args, model, device, logger)

    # Training loop
    write_log(logger, f"Start training from epoch {start_epoch}")
    for epoch_idx in range(start_epoch, args.num_epochs):
        # Training
        train_metrics = trainer.train_epoch(
            dataloader_dict['train'], optimizer, scheduler, cls_loss, epoch_idx
        )

        # Validation
        valid_metrics = trainer.validate(dataloader_dict['valid'], cls_loss, epoch_idx)

        # Scheduler step (epoch-level)
        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_metrics['loss'])

        # Determine objective value
        valid_objective_value = trainer.get_objective_value(valid_metrics)

        # Save best checkpoint
        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            write_log(logger, f"VALID - Saving checkpoint for best valid {args.optimize_objective}...")
            early_stopping_counter = 0

            check_path(checkpoint_dir)

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

        # Save last checkpoint
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
                'TRAIN/Epoch_Loss': train_metrics['loss'],
                'TRAIN/Epoch_Acc': train_metrics['accuracy'],
                'TRAIN/Epoch_F1': train_metrics['f1'],
                'VALID/Epoch_Loss': valid_metrics['loss'],
                'VALID/Epoch_Acc': valid_metrics['accuracy'],
                'VALID/Epoch_F1': valid_metrics['f1'],
                'Learning_Rate': optimizer.param_groups[0]['lr'],
                'Epoch_Index': epoch_idx
            })
            wandb_mgr.alert(
                title='Epoch End',
                text=f"VALID - Epoch {epoch_idx} - Loss: {valid_metrics['loss']:.4f} - Acc: {valid_metrics['accuracy']:.4f}",
                level='INFO',
                wait_duration=300
            )

        # Early stopping check
        if early_stopping_counter >= args.early_stopping_patience:
            write_log(logger, f"VALID - Early stopping at epoch {epoch_idx}...")
            break

    write_log(logger, f"Done! Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")

    # Save final model (copy best checkpoint to final_model.pt)
    final_model_path = get_model_path(args)
    check_path(os.path.dirname(final_model_path))
    shutil.copyfile(os.path.join(checkpoint_dir, 'checkpoint.pt'), final_model_path)
    write_log(logger, f"FINAL - Saved final model to {final_model_path}")

    # Log training summary
    try:
        pifi_logger = PiFiLogger.from_args(args)
        pifi_logger.log_training_summary(
            best_epoch=best_epoch_idx,
            best_metric=abs(best_valid_objective_value),
            best_metric_name=args.optimize_objective,
            total_epochs=epoch_idx + 1
        )
    except Exception as e:
        write_log(logger, f"Failed to save training summary: {e}")

    # Finish W&B run
    if args.use_wandb and wandb_mgr:
        wandb_mgr.finish()
