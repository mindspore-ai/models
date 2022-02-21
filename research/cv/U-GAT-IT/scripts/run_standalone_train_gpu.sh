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

if [[ $# -le 1 ]]; then
    echo "Usage: bash run_standalone_train_gpu.sh \
    [DEVICE_ID] [DATA_PATH] [<LR>] [<LIGHT>] \
    [<LOSS_SCALE>] [<USE_GLOBAL_NORM>]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DEVICE_ID=$1
DATA_PATH=$(get_real_path $2)
LR=${3:-"0.0001"}
LIGHT=${4:-"True"}
LOSS_SCALE=${5:-"1.0"}
USE_GLOBAL_NORM=${6:-"False"}

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a dir"
exit 1
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../src/default_config.yaml"

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log

python train.py \
    --device_target GPU \
    --device_id $DEVICE_ID \
    --data_path $DATA_PATH \
    --light $LIGHT \
    --lr $LR \
    --loss_scale $LOSS_SCALE \
    --use_global_norm $USE_GLOBAL_NORM \
    --config_path $CONFIG_FILE &> log &

cd ..
