#!/usr/bin/env bash
# OAR job: build the evaluation pool and the EUNIS legend.
# Usage: oarsub -n georeset-prepare -l /nodes=1/core=8,walltime=04:00:00 -t night \
#          "bash scripts/g5k/prepare.sh <work_dir>"
set -euo pipefail

WORK="${1:?usage: prepare.sh <work_dir>}"
cd "$HOME/GeoReSeT-Corpus-Assessment"
.venv/bin/georeset prepare --work "$WORK"
