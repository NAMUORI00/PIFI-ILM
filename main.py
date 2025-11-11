#!/usr/bin/env python3
"""
PiFi: Plug-in and Fine-tuning
Unified entry point for all tasks (classification, entailment, etc.)

Usage:
    # Classification
    python main.py --task classification --job training --task_dataset sst2 --method pifi

    # Entailment
    python main.py --task entailment --job training --task_dataset mnli --method pifi

    # With ILM auto-selection
    python main.py --task classification --job training --task_dataset sst2 \
        --method pifi --llm_model llama3.1 --auto_select_layer true
"""

import time
import sys
from core import ArgParser, set_random_seed, check_path, run_pipeline


def main():
    """Main entry point for PiFi"""
    # Parse arguments
    parser = ArgParser()
    args = parser.get_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        set_random_seed(args.seed)

    # Start timer
    start_time = time.time()

    # Check required paths (can be customized per project)
    for path in []:
        check_path(path)

    # Run the pipeline
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'Completed {args.job} for {args.task} task')
    print(f'Time elapsed: {elapsed_time / 60:.2f} minutes')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
