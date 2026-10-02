#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/unsloth-nvfp4-env"

if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$PYTHON_VERSION" != "3.13" ]]; then
    echo "Existing environment uses Python $PYTHON_VERSION; expected Python 3.13: $VENV_DIR" >&2
    exit 1
  fi
  echo "Python 3.13 environment already exists at $VENV_DIR; leaving it unchanged."
  exit 0
fi

if [[ -e "$VENV_DIR" ]]; then
  echo "Environment path exists but is not a usable virtual environment: $VENV_DIR" >&2
  exit 1
fi

PYTHON_BIN="$(command -v python3.13 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.13 is required. Install it, then rerun this script." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
echo "Created Python 3.13 environment at $VENV_DIR"
echo "Activate it with: source \"$VENV_DIR/bin/activate\""
echo "Install a CUDA-compatible PyTorch, vLLM, and FlashInfer build for this host."
echo "Configure Hugging Face access separately; this script does not install packages or manage credentials."