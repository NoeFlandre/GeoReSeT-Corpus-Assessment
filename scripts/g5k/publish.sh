#!/usr/bin/env bash
# OAR job: upload the Parquet files of a results folder to Hugging Face.
# The write token lives in a file only its owner can read; it never enters the repo.
# Usage: oarsub -n georeset-publish -l /nodes=1/core=1,walltime=00:30:00 \
#          "bash scripts/g5k/publish.sh <out_dir>"
set -euo pipefail

OUT="${1:?usage: publish.sh <out_dir>}"
cd "$HOME/GeoReSeT-Corpus-Assessment"
HF_TOKEN="$(cat "$HOME/.config/georeset/hf_token")" .venv/bin/georeset publish --out "$OUT"
