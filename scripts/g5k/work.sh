#!/usr/bin/env bash
# OAR worker job: run several worker processes on one GPU. Each process claims shards
# from this site's queue, so the processes share the work without duplicates.
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
MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
if [ "$MEM_MIB" -ge 70000 ]; then PROCS=3; else PROCS=2; fi

for _ in $(seq "$PROCS"); do
  .venv/bin/georeset work --work "$WORK" --out "$OUT" --limit "$LIMIT" \
    --shards "$SHARDS" --budget-minutes "$BUDGET" --estimate-minutes "$EST" \
    --gpu "$GPU" --owner "$OWNER" --owners "$OWNERS" &
  sleep 20
done
wait
