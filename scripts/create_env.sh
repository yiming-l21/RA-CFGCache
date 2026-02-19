#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3.10}
VENV_DIR=${VENV_DIR:-.venv}

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install --no-cache-dir --force-reinstall opencv-python-headless==4.13.0.92

echo "[INFO] Environment ready: $VENV_DIR"
