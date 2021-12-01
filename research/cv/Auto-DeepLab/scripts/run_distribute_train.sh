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
if [ $# != 4 ]
then
  echo "=========================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_distribute_train.sh [RANK_SIZE] [RANK_TABLE_FILE] [DATASET_PATH] [EPOCHS]"
  echo "For example:"
  echo "cd /PATH/TO/Auto-DeepLab"
  echo "bash /PATH/TO/Auto-DeepLab/scripts/run_distribute_train.sh \
        8  /PATH/TO/RANK_TABLE_FILE/hccl_8p_01234567_127.0.0.1.json \
        /PATH/TO/cityscapes/  4000"
  echo "Using absolute path is recommended"
  echo "=========================================================================================="
  exit 1
fi

ulimit -c unlimited
export SLOG_PRINT_TO_STDOUT=0
export HCCL_CONNECT_TIMEOUT=480
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2
export RANK_START_ID=0
export DATASET_PATH=$3
export EPOCHS=$4
TRAIN_CODE_PATH=$(pwd)
OUTPUT_PATH=${TRAIN_CODE_PATH}/OUTPUTS

if [ -d "${OUTPUT_PATH}" ]; then
  echo "${OUTPUT_PATH} already exists"
  exit 1
fi
mkdir -p "${OUTPUT_PATH}"
mkdir "${OUTPUT_PATH}"/ckpt

echo 'here'
for((i=0; i<RANK_SIZE; i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='"${i}"', device id='${DEVICE_ID}'...'
    mkdir "${OUTPUT_PATH}"/device${DEVICE_ID}
    cd "${OUTPUT_PATH}"/device${DEVICE_ID} || exit
    python "${TRAIN_CODE_PATH}"/train.py --out_path="${OUTPUT_PATH}"/ckpt \
                                         --data_path="${DATASET_PATH}" \
                                         --modelArts=False \
                                         --epochs="${EPOCHS}" \
                                         --bn_momentum=0.995 >log.txt 2>&1 &
done
