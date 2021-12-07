#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
if [ $# != 3 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_train.sh [DEVICE_ID] [DATASET_PATH] [EPOCHS]"
  echo "For example:"
  echo "cd Auto-DeepLab"
  echo "bash /code/Auto-DeepLab/scripts/run_train.sh \
        0 /data/cityscapes/ 4000"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

ulimit -c unlimited
export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0
export DATASET_PATH=$2
export EPOCHS=$3
TRAIN_CODE_PATH=$(pwd)
OUTPUT_PATH=${TRAIN_CODE_PATH}/OUTPUTS

if [ -d "${OUTPUT_PATH}" ]; then
  echo "${OUTPUT_PATH} already exists"
  exit 1
fi
mkdir -p "${OUTPUT_PATH}"
mkdir "${OUTPUT_PATH}"/device"${DEVICE_ID}"
mkdir "${OUTPUT_PATH}"/ckpt
cd "${OUTPUT_PATH}"/device"${DEVICE_ID}" || exit

python "${TRAIN_CODE_PATH}"/train.py --out_path="${OUTPUT_PATH}"/ckpt \
                                      --data_path="${DATASET_PATH}" \
                                      --modelArts=False \
                                      --parallel=False \
                                      --batch_size=8 \
                                      --affine=False \
                                      --epochs="${EPOCHS}" >log.txt 2>&1 &

