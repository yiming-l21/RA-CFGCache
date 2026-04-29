#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="racfgcache"
PYTHON_VERSION="3.10"

if ! conda env list | grep -q "^$ENV_NAME"; then
    echo "[INFO] 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "[INFO] 激活环境: $ENV_NAME"
source activate "$ENV_NAME"

python -m pip install -U pip setuptools wheel

python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install --no-cache-dir --force-reinstall opencv-python-headless==4.13.0.92

echo "[INFO] Environment ready: $ENV_NAME"
echo "[INFO] 已成功切换到 conda 环境: $ENV_NAME"