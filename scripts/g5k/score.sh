#!/usr/bin/env bash
# OAR job: score the shards of a results folder at every EUNIS level.
# Usage: oarsub -n georeset-score -l /nodes=1/core=1,walltime=00:15:00 \
#          "bash scripts/g5k/score.sh <work_dir> <out_dir>"
set -euo pipefail

WORK="${1:?usage: score.sh <work_dir> <out_dir>}"
OUT="${2:?usage: score.sh <work_dir> <out_dir>}"
cd "$HOME/GeoReSeT-Corpus-Assessment"
.venv/bin/georeset score --work "$WORK" --out "$OUT"
