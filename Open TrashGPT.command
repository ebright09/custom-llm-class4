#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Setting up TrashGPT's local Python environment…"
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import torch, numpy' >/dev/null 2>&1; then
  echo "Installing local model dependencies. No model API is used."
  .venv/bin/python -m pip install 'torch>=2.2,<3' 'numpy>=1.26,<3'
fi
.venv/bin/python trashgpt.py --open
