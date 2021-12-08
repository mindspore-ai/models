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
if [ $# != 8 ]
then
  echo "=========================================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_distribute_train.sh [RANK_SIZE] [RANK_TABLE_FILE] [MODEL_NAME ] [BATCH_SIZE] [EPOCH_SIZE]
                                             [LEARNING_RATE] [DECAY_STEPS] [DATA_ROOT]"
  echo "For example:"
  echo "cd /PATH/TO/DAM"
  echo "bash ./scripts/run_distribute_train.sh 8 ABSOLUTE_PATH/rank_table_8pcs.json DAM_ubuntu 256 2 0.001 400
                                            ABSOLUTE_PATH/DAM/data/ubuntu/"
  echo "Using absolute path is recommended"
  echo "=========================================================================================="
  exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in OUTPUT_PATH/devicex/log.txt"

ulimit -c unlimited
export SLOG_PRINT_TO_STDOUT=0
export RANK_SIZE=$1
export RANK_TABLE_FILE=$2
export RANK_START_ID=0
TRAIN_CODE_PATH=$(pwd)
OUTPUT_PATH=${TRAIN_CODE_PATH}/OUTPUTS_8p_2

if [ -d "${OUTPUT_PATH}" ]; then
  echo "${OUTPUT_PATH} already exists"
  exit 1
fi
mkdir -p "${OUTPUT_PATH}"
mkdir "${OUTPUT_PATH}"/ckpt

for((i=0; i<RANK_SIZE; i++));
do
    export RANK_ID=${i}
    export DEVICE_ID=$((i + RANK_START_ID))
    echo 'start rank='"${i}"', device id='${DEVICE_ID}'...'
    mkdir "${OUTPUT_PATH}"/device${DEVICE_ID}
    cd "${OUTPUT_PATH}"/device${DEVICE_ID} || exit
    python "${TRAIN_CODE_PATH}"/train.py  --parallel=True \
                                          --model_name=$3 \
                                          --is_emb_init=True \
                                          --batch_size=$4 \
                                          --epoch_size=$5 \
                                          --learning_rate=$6 \
                                          --decay_steps=$7 \
                                          --data_root=$8 \
                                          --output_path="${OUTPUT_PATH}"/ckpt >log.txt 2>&1 &
done
