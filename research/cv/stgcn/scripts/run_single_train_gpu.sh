#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================

if [ $# != 4 ]; then
  echo "Usage: bash scripts/run_single_train_gpu.sh [DATA_PATH] [N_PRED] [GRAPH_CONV_TYPE] [DEVICE_ID]"
  echo: "Example: bash scripts/run_single_train_gpu.sh ./data/pemsd7-m 9 chebconv 0"
  exit 1
fi


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}


DATA_PATH=$(get_real_path $1)
echo $DATA_PATH


if [ ! -d $DATA_PATH ]
then
    echo "error: train_code_path=$DATA_PATH is not a dictionary."
exit 1
fi

ulimit -c unlimited
export SLOG_PRINT_TO_STDOUT=0
export N_PRED=$2
export GRAPH_CONV_TYPE=$3
export RANK_SIZE=1
export DEVICE_ID=$4

rm -rf ./train$DEVICE_ID
mkdir ./train$DEVICE_ID
cp ./*.py ./train$DEVICE_ID
cp -r ./src ./train$DEVICE_ID
cd ./train$DEVICE_ID || exit

echo "start training on GPU device id $DEVICE_ID"
python train.py \
       --device_target="GPU"  \
       --epochs=50 \
       --run_distribute=False   \
       --device_id=$DEVICE_ID  \
       --data_url=${DATA_PATH}   \
       --train_url=./checkpoint   \
       --run_modelarts=False \
       --n_pred=$N_PRED     \
       --graph_conv_type=$GRAPH_CONV_TYPE > train.log 2>&1 &
cd ..
