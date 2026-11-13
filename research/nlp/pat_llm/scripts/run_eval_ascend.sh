#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: bash run_eval_ascend.sh DATA_DIR LLM_DIR CKPT [DATA_PATH] [TARGET]"
  exit 1
fi

DATA_DIR=$(realpath "$1")
LLM_DIR=$(realpath "$2")
CHECKPOINT=$(realpath "$3")
DATA_PATH=${4:-C5/C5-2.csv}
TARGET=${5:-C5}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "${PROJECT_DIR}"
python eval.py \
  --config_path default_config.yaml \
  --set "root_path=${DATA_DIR}" \
  --set "data_path=${DATA_PATH}" \
  --set "target=${TARGET}" \
  --set "llm_model_path=${LLM_DIR}" \
  --set "checkpoint_path=${CHECKPOINT}"
