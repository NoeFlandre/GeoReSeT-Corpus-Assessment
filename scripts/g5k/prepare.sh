#!/usr/bin/env bash
# OAR job step for building the evaluation pool in small pieces.
# Usage:
#   prepare.sh <work_dir> labels
#   prepare.sh <work_dir> part <index> <parts>
#   prepare.sh <work_dir> merge <parts>
# Hugging Face downloads go to node-local /tmp so the home quota is not used.
set -euo pipefail

WORK="${1:?usage: prepare.sh <work_dir> labels|part|merge ...}"
STEP="${2:?usage: prepare.sh <work_dir> labels|part|merge ...}"
export HF_HOME="/tmp/hf-georeset-${OAR_JOB_ID:-local}"
cd "$HOME/GeoReSeT-Corpus-Assessment"

case "$STEP" in
  labels) .venv/bin/georeset prepare-labels --work "$WORK" ;;
  part)   .venv/bin/georeset prepare-part --work "$WORK" --part "${3:?part index}" --parts "${4:?parts}" ;;
  merge)  .venv/bin/georeset prepare-merge --work "$WORK" --parts "${3:?parts}" ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
rm -rf "$HF_HOME"
