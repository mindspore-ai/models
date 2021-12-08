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
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_train.sh [DEVICE_ID] [MODEL_NAME ] [BATCH_SIZE] [EPOCH_SIZE] [LEARNING_RATE] [DECAY_STEPS] [DATA_ROOT] [OUTPUT_PATH]"
  echo "For example:"
  echo "bash ./scripts/run_train.sh 0 DAM_ubuntu 256 2 0.001 400 /ABSOLUTE_PATH/data/ubuntu/ /ABSOLUTE_PATH/output/ubuntu/"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in ./train.log"

ulimit -c unlimited
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$1
echo "start training for device $DEVICE_ID"
python train.py --model_name=$2 \
                --batch_size=$3 \
                --is_emb_init=True \
                --epoch_size=$4 \
                --learning_rate=$5 \
                --decay_steps=$6 \
                --data_root=$7 \
                --output_path=$8 >train.log 2>&1 &