#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: bash run_standalone_train_ascend.sh DATA_DIR LLM_DIR [DATA_PATH] [TARGET]"
  exit 1
fi

DATA_DIR=$(realpath "$1")
LLM_DIR=$(realpath "$2")
DATA_PATH=${3:-C5/C5-2.csv}
TARGET=${4:-C5}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "${PROJECT_DIR}"
python train.py \
  --config_path default_config.yaml \
  --set "root_path=${DATA_DIR}" \
  --set "data_path=${DATA_PATH}" \
  --set "target=${TARGET}" \
  --set "llm_model_path=${LLM_DIR}"
