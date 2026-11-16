#!/usr/bin/env bash
# One-time environment setup on a Grid'5000 frontend. Run from the repo root.
# The frontend only does light work, so this step only installs packages.
set -euo pipefail

uv sync --frozen --python 3.12
uv pip install --python .venv/bin/python "vllm==0.33.0"
.venv/bin/georeset --help > /dev/null
echo "environment ready in $(pwd)/.venv"
