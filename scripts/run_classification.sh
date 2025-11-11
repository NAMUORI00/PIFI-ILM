#!/usr/bin/env bash
# Classification-only wrapper for run_experiments.sh

TASKS="classification" bash "$(dirname "$0")/run_experiments.sh" "$@"
