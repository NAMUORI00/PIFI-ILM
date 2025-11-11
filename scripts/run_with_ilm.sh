#!/usr/bin/env bash
# Run experiments with ILM auto-selection enabled

USE_ILM=true bash "$(dirname "$0")/run_experiments.sh" "$@"
