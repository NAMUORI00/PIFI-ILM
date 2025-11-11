#!/usr/bin/env bash
# Entailment-only wrapper for run_experiments.sh

TASKS="entailment" bash "$(dirname "$0")/run_experiments.sh" "$@"
