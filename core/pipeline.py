"""
PiFi Pipeline
Task dispatcher and ILM integration
"""

import argparse
import os
import sys
from typing import Callable


def get_job_function(args: argparse.Namespace) -> Callable:
    """
    Get the appropriate job function based on task and job type.

    Returns:
        Callable function that executes the specified job
    """
    if args.job is None:
        raise ValueError('Please specify the job to do using --job argument.')

    task = args.task
    job = args.job

    # Ensure legacy package roots are importable for internal absolute imports
    # Legacy modules use 'from utils...' style expecting Classification/ or TextualEntailment/ on sys.path
    try:
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        _cls = os.path.join(_root, 'Classification')
        _ent = os.path.join(_root, 'TextualEntailment')
        for _p in (_cls, _ent):
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.append(_p)
    except Exception:
        pass

    # Classification task
    if task == 'classification':
        if job == 'preprocessing':
            from tasks.classification.task.classification.preprocessing import preprocessing as job_func
        elif job in ['training', 'resume_training']:
            from tasks.classification.task.classification.train import training as job_func
        elif job == 'testing':
            from tasks.classification.task.classification.test import testing as job_func
        else:
            raise ValueError(f'Invalid job for classification: {job}')

    # Entailment task
    elif task == 'entailment':
        if job == 'preprocessing':
            from tasks.entailment.task.entailment.preprocessing import preprocessing as job_func
        elif job in ['training', 'resume_training']:
            from tasks.entailment.task.entailment.train import training as job_func
        elif job == 'testing':
            from tasks.entailment.task.entailment.test import testing as job_func
        else:
            raise ValueError(f'Invalid job for entailment: {job}')

    else:
        raise ValueError(f'Invalid task: {task}')

    return job_func


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Main pipeline execution function.
    Handles ILM auto-selection if enabled, then runs the specified job.

    Args:
        args: Parsed command-line arguments
    """
    # ILM auto layer selection (if enabled and using PiFi method for training)
    if (args.method == 'pifi' and
        args.auto_select_layer and
        args.job in ['training', 'resume_training']):

        print(f"[Pipeline] ILM auto-selection enabled for {args.task}/{args.task_dataset}")
        print(f"[Pipeline] Selection parameters: samples={args.selection_samples}, "
              f"pcs={args.selection_pcs}, top_pc={args.selection_top_pc}")

        from selection import auto_select_layer
        selected_layer = auto_select_layer(args)

        # Update args with selected layer
        args.layer_num = selected_layer
        print(f"[Pipeline] Selected layer: {selected_layer}")

    # Get and execute the job function
    job_func = get_job_function(args)

    print(f"[Pipeline] Starting {args.job} for {args.task} task on {args.task_dataset} dataset")
    job_func(args)
    print(f"[Pipeline] Completed {args.job} for {args.task} task")
