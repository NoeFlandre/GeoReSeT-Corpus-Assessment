#!/usr/bin/env bash
# OAR job: label sentences on one GPU, writing resumable Parquet shards.
# Usage: oarsub -n georeset-run -l gpu=1,walltime=01:00:00 \
#          "bash scripts/g5k/run.sh <work_dir> <out_dir> <limit> [batch_size]"
set -euo pipefail

WORK="${1:?usage: run.sh <work_dir> <out_dir> <limit> [batch_size]}"
OUT="${2:?usage: run.sh <work_dir> <out_dir> <limit> [batch_size]}"
LIMIT="${3:?usage: run.sh <work_dir> <out_dir> <limit> [batch_size]}"
BATCH="${4:-500}"

cd "$HOME/GeoReSeT-Corpus-Assessment"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
.venv/bin/georeset run --work "$WORK" --out "$OUT" --limit "$LIMIT" \
  --batch-size "$BATCH" --gpu "$GPU"
