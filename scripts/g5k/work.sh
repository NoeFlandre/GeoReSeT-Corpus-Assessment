#!/usr/bin/env bash
# OAR worker job: claim and run shards from this site's queue within a time budget.
# Usage: work.sh <work_dir> <out_dir> <limit> <shards> <budget_min> <owner> <owners> [estimate_min]
set -euo pipefail

WORK="${1:?usage: work.sh <work_dir> <out_dir> <limit> <shards> <budget_min> <owner> <owners>}"
OUT="${2:?out_dir}"
LIMIT="${3:?limit}"
SHARDS="${4:?shards}"
BUDGET="${5:?budget_min}"
OWNER="${6:?owner}"
OWNERS="${7:?owners}"
EST="${8:-8}"

cd "$HOME/GeoReSeT-Corpus-Assessment"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
.venv/bin/georeset work --work "$WORK" --out "$OUT" --limit "$LIMIT" \
  --shards "$SHARDS" --budget-minutes "$BUDGET" --estimate-minutes "$EST" \
  --gpu "$GPU" --owner "$OWNER" --owners "$OWNERS"
