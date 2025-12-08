"""
Unified Argument Parser for PiFi
Supports all tasks: classification, entailment, etc.
"""

import os
import argparse


def parse_bool(value: str):
    """Parse boolean string to bool"""
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PiFi: Plug-in and Fine-tuning')
        try:
            self.user_name = os.getlogin()
        except OSError:
            # Fallback in containerized/daemonized environments with no controlling TTY
            self.user_name = os.environ.get('USER') or os.environ.get('USERNAME') or 'user'

        # Task arguments
        task_list = ['classification', 'entailment']
        self.parser.add_argument('--task', type=str, choices=task_list, default='classification',
                                 help='Task to do; Default is "classification"')

        job_list = ['preprocessing', 'training', 'resume_training', 'testing']
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Default is "training"')

        # All datasets (classification + entailment)
        dataset_list = [
            # Classification datasets
            'imdb', 'sst2', 'cola', 'trec', 'subj', 'agnews', 'mr', 'cr',
            'proscons', 'dbpedia', 'yelp_polarity', 'tweet_offensive',
            'tweet_sentiment_binary', 'yelp_full', 'yahoo_answers_title',
            'yahoo_answers_full', 'nsmc', 'filmstarts', 'chinese_toxicity',
            # Entailment datasets
            'snli', 'mnli', 'amazon_polarity', 'squad'
        ]
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='sst2',
                                 help='Dataset for the task; Default is "sst2"')
        self.parser.add_argument('--test_dataset', type=str, choices=dataset_list, default='sst2',
                                 help='Test dataset for the task; Default is same as task_dataset')

        self.parser.add_argument('--description', type=str, default='default',
                                 help='Description of the experiment; Default is "default"')

        method_list = ['base', 'pifi']
        self.parser.add_argument('--method', type=str, choices=method_list, default='pifi',
                                 help='Method to use; Default is "pifi"')

        # Path arguments
        # Unified path structure (v3):
        #   --data_path    : Raw datasets (external, read-only)
        #   --cache_path   : HuggingFace model cache (.cache/)
        #   --logs_path    : All experiment artifacts (logs/)
        #                    ├── preprocessed/   (preprocessed data)
        #                    └── experiments/    (checkpoints, models, results)
        #
        # Legacy paths (deprecated): --preprocess_path, --model_path, --checkpoint_path, --result_path
        is_docker = os.path.exists('/.dockerenv') or os.environ.get('PIFI_IN_DOCKER') == '1'
        default_root = '/app' if is_docker else f'/nas_homes/{self.user_name}'

        def _env_or(key: str, default: str) -> str:
            return os.environ.get(key, default)

        # Primary paths (v3)
        data_path_def = _env_or('PIFI_DATA_PATH', f'{default_root}/dataset' + ('/' if not default_root.endswith('/') else ''))
        cache_path_def = _env_or('PIFI_CACHE_PATH', f'{default_root}/.cache' if is_docker else f'/nas_homes/{self.user_name}/.cache')
        # logs_path: Empty default to allow result_path backward compat; actual default set in get_args()
        logs_path_def = _env_or('PIFI_LOGS_PATH', '')

        # Legacy paths (deprecated, for backward compatibility)
        preprocess_path_def = _env_or('PIFI_PREPROCESS_PATH', '')  # Empty = use logs_path/preprocessed
        model_path_def = _env_or('PIFI_MODEL_PATH', '')  # Empty = use logs_path/experiments
        checkpoint_path_def = _env_or('PIFI_CHECKPOINT_PATH', '')  # Empty = use logs_path/experiments
        result_path_def = _env_or('PIFI_RESULT_PATH', '')  # Empty = alias for logs_path

        # Primary path arguments
        self.parser.add_argument('--data_path', type=str, default=data_path_def,
                                 help='Path to the raw dataset before preprocessing')
        self.parser.add_argument('--cache_path', type=str, default=cache_path_def,
                                 help='Path to HuggingFace model cache (.cache/)')
        self.parser.add_argument('--logs_path', type=str, default=logs_path_def,
                                 help='Path to all experiment artifacts (preprocessed/, experiments/)')

        # Legacy path arguments (deprecated)
        self.parser.add_argument('--preprocess_path', type=str, default=preprocess_path_def,
                                 help='[DEPRECATED] Use --logs_path instead. Preprocessed data saved to logs/preprocessed/')
        self.parser.add_argument('--model_path', type=str, default=model_path_def,
                                 help='[DEPRECATED] Use --logs_path instead. Models saved to logs/experiments/')
        self.parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path_def,
                                 help='[DEPRECATED] Use --logs_path instead. Checkpoints saved to logs/experiments/')
        self.parser.add_argument('--result_path', type=str, default=result_path_def,
                                 help='[DEPRECATED] Alias for --logs_path')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default='PiFi',
                                 help='Name of the project; Default is "PiFi"')

        model_type_list = [
            'bert', 'bert_large', 'modern_bert', 'smollm', 'roberta', 'albert',
            'electra', 'deberta', 'debertav3', 'roberta-large', 'kcbert',
            'mbert', 'lstm', 'gru', 'rnn'
        ]
        self.parser.add_argument('--model_type', type=str, choices=model_type_list, default='bert',
                                 help='Type of the model to use; Default is "bert"')

        llm_model_list = [
            'llama2', 'llama3', 'llama3.1', 'llama3.1_instruct',
            'mistral0.1', 'mistral0.3',
            'qwen2_7b', 'qwen2_0.5b', 'qwen2_1.5b', 'qwen2_72b',
            'gemma2', 'falcon',
            'kollama', 'gerllama', 'chillama'
        ]
        self.parser.add_argument('--llm_model', type=str, choices=llm_model_list, default='llama3.1',
                                 help='LLM model to use; Default is "llama3.1"')
        # Backward-compat alias
        self.parser.add_argument('--llm', dest='llm_model', type=str, choices=llm_model_list,
                                 help='Alias for --llm_model')

        self.parser.add_argument('--model_ispretrained', type=parse_bool, default=True,
                                 help='Whether to use pretrained model; Default is True')
        self.parser.add_argument('--rnn_isbidirectional', type=parse_bool, default=True,
                                 help='Whether to use bidirectional RNNs; Default is True')
        self.parser.add_argument('--min_seq_len', type=int, default=4,
                                 help='Minimum sequence length of the input; Default is 4')
        self.parser.add_argument('--max_seq_len', type=int, default=100,
                                 help='Maximum sequence length of the input; Default is 100')
        self.parser.add_argument('--dropout_rate', type=float, default=0.2,
                                 help='Dropout rate of the model; Default is 0.2')
        self.parser.add_argument('--padding', type=str, default='cls',
                                 help='Padding method of the input; Default is "cls"')
        self.parser.add_argument('--freeze', type=parse_bool, default=True,
                                 help='Freeze LLM layers during training; Default is True')

        # Model - Size arguments
        self.parser.add_argument('--embed_size', type=int, default=768,
                                 help='Embedding size of the model; Default is 768')
        self.parser.add_argument('--hidden_size', type=int, default=768,
                                 help='Hidden size of the model; Default is 768')
        self.parser.add_argument('--num_layers_rnn', type=int, default=2,
                                 help='Number of layers of RNNs; Default is 2')
        self.parser.add_argument('--num_layers_transformer', type=int, default=6,
                                 help='Number of layers of Transformer Encoder; Default is 6')
        self.parser.add_argument('--num_heads_transformer', type=int, default=8,
                                 help='Number of heads of Transformer Encoder; Default is 8')
        self.parser.add_argument('--layer_num', type=int, default=-1,
                                 help='Layer number of the LLM model; Default is -1 (last layer)')

        # Auto layer selection (ILM)
        self.parser.add_argument('--auto_select_layer', type=parse_bool, default=False,
                                 help='Automatically select LLM layer before training using ILM method; Default is False')
        self.parser.add_argument('--selection_samples', type=int, default=400,
                                 help='Number of samples for layer selection; Default is 400')
        self.parser.add_argument('--selection_pcs', type=int, default=16,
                                 help='Number of principal components per layer for selection; Default is 16')
        self.parser.add_argument('--selection_top_pc', type=int, default=5,
                                 help='Top PCs by label-correlation to patch; Default is 5')
        self.parser.add_argument('--selection_layer_stride', type=int, default=1,
                                 help='Stride for layer scoring (e.g., 2 scores every other layer); Default is 1')
        self.parser.add_argument('--selection_patch_lambda', type=str, default='0.5,1.0',
                                 help='Comma-separated lambdas for PC patching strength (e.g., "0.5,1.0"); Default is "0.5,1.0"')
        self.parser.add_argument('--selection_pooling', type=str, choices=['first','mean'], default='mean',
                                 help='Pooling for hidden states when scoring layers: first token (CLS) or mean; Default is mean')
        self.parser.add_argument('--selection_split', type=str, choices=['train','validation','test'], default='validation',
                                 help='Dataset split used for selection (task-specific mapping applied); Default is validation')
        self.parser.add_argument('--selection_max_length', type=int, default=128,
                                 help='Max token length for selection encoding (set 0 to inherit --max_seq_len)')
        self.parser.add_argument('--selection_dtype', type=str, choices=['fp16','fp32'], default='fp16',
                                 help='LLM dtype for selection forward pass; Default is fp16 (uses fp32 on CPU)')
        self.parser.add_argument('--selection_stratified', type=parse_bool, default=True,
                                 help='Use stratified sampling for selection set; Default is True')
        self.parser.add_argument('--selection_score_mode', type=str, choices=['mixed','probe','fisher','silhouette','robust'], default='mixed',
                                 help='Scoring mode for ILM selection; Default is mixed')
        self.parser.add_argument('--selection_score_alpha', type=float, default=0.4,
                                 help='Weight for probe score when mode=mixed; Default is 0.4')
        self.parser.add_argument('--selection_score_beta', type=float, default=0.3,
                                 help='Weight for fisher score when mode=mixed; Default is 0.3')
        self.parser.add_argument('--selection_score_gamma', type=float, default=0.3,
                                 help='Weight for silhouette score when mode=mixed; Default is 0.3')
        self.parser.add_argument('--selection_depth_bias', type=float, default=0.3,
                                 help='Depth penalty factor to de-bias deep layers; Default is 0.3')

        # Robust selection parameters (K-fold CV + multi-seed + selectivity)
        self.parser.add_argument('--robust_selection', type=parse_bool, default=False,
                                 help='Use robust probe with K-fold CV, multi-seed averaging, and selectivity score; Default is False')
        self.parser.add_argument('--selection_n_folds', type=int, default=5,
                                 help='Number of CV folds for robust selection; Default is 5')
        self.parser.add_argument('--selection_n_seeds', type=int, default=3,
                                 help='Number of seeds for multi-seed averaging in robust selection; Default is 3')
        self.parser.add_argument('--selection_use_confidence_weight', type=parse_bool, default=False,
                                 help='Apply confidence weighting in robust mode (empirical heuristic); Default is False (pure selectivity)')

        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR',
                         'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='Adam',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--scheduler', type=str, choices=scheduler_list, default='None',
                                 help="Scheduler to use; If None, no scheduler is used; Default is None")

        # Training arguments
        self.parser.add_argument('--num_epochs', type=int, default=3,
                                 help='Training epochs; Default is 3')
        self.parser.add_argument('--learning_rate', type=float, default=5e-5,
                                 help='Learning rate of optimizer; Default is 5e-5')
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Batch size; Default is 32')
        self.parser.add_argument('--weight_decay', type=float, default=0,
                                 help='Weight decay; Default is 0; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=5,
                                 help='Gradient clipping norm; Default is 5')
        self.parser.add_argument('--label_smoothing_eps', type=float, default=0.05,
                                 help='Label smoothing epsilon; Default is 0.05')
        self.parser.add_argument('--early_stopping_patience', type=int, default=5,
                                 help='Early stopping patience; No early stopping if None; Default is 5')
        self.parser.add_argument('--train_valid_split', type=float, default=0.2,
                                 help='Train/Valid split ratio; Default is 0.2')

        objective_list = ['loss', 'accuracy', 'f1']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='accuracy',
                                 help='Objective to optimize; Default is accuracy')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=16, type=int,
                                 help='Batch size for test; Default is 16')

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type=int, default=2023,
                                 help='Random seed; Default is 2023')
        self.parser.add_argument('--use_wandb', type=parse_bool, default=True,
                                 help='Using wandb for experiment tracking; Default is True')
        self.parser.add_argument('--log_selection', type=parse_bool, default=True,
                                 help='Log ILM selection results to W&B when enabled; Default is True')
        self.parser.add_argument('--log_selection_pca', type=parse_bool, default=True,
                                 help='Log per-layer PCA scatter plots during ILM selection; Default is True')
        self.parser.add_argument('--selection_plot_layers', type=str, default='best,first,mid,last',
                                 help='Layers to plot PCA scatter: comma list (e.g., "best,first,last,3,7") or "all"; Default best,first,mid,last')
        self.parser.add_argument('--selection_plot_max_layers', type=int, default=6,
                                 help='Max number of layers to plot PCA scatter; Default is 6')
        self.parser.add_argument('--save_selection_plots', type=parse_bool, default=False,
                                 help='Save ILM selection plots to disk (result_path/logs/); Default is False')
        self.parser.add_argument('--log_test_local', type=parse_bool, default=True,
                                 help='Save test results to local JSON (result_path/logs/); Default is True')
        self.parser.add_argument('--log_freq', default=500, type=int,
                                 help='Logging frequency; Default is 500')

    def get_args(self):
        """Parse and return arguments"""
        args = self.parser.parse_args()

        # Auto-update proj_name based on task
        if args.proj_name == 'PiFi':
            if args.task == 'classification':
                args.proj_name = 'PiFi_Classification'
            elif args.task == 'entailment':
                args.proj_name = 'PiFi_TextualEntailment'

        # Update test_dataset if not specified
        if not hasattr(args, 'test_dataset') or args.test_dataset == 'sst2':
            args.test_dataset = args.task_dataset

        # Backward compatibility: result_path → logs_path alias
        if args.result_path and not args.logs_path:
            args.logs_path = args.result_path
        elif not args.logs_path:
            # Set default if neither is set
            is_docker = os.path.exists('/.dockerenv') or os.environ.get('PIFI_IN_DOCKER') == '1'
            default_root = '/app' if is_docker else f'/nas_homes/{self.user_name}'
            args.logs_path = f'{default_root}/logs' if is_docker else f'/nas_homes/{self.user_name}/logs/PiFi'

        # Backward compatibility: Set result_path as alias to logs_path
        if not args.result_path:
            args.result_path = args.logs_path

        return args
