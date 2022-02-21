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
    echo "Usage: bash run_distributed_train_gpu.sh \
    [RANK_SIZE] [DATA_PATH] [<LR>] [<LIGHT>] \
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

export RANK_SIZE=$1
DATA_PATH=$(get_real_path $2)
LR=${3:-"0.000025"}
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

if [ -d "run_distribute_train" ];
then
    rm -rf ./run_distribute_train
fi
mkdir run_distribute_train
cp -r ../src/ ../train.py ./run_distribute_train
cd run_distribute_train || exit

echo "start training on $RANK_SIZE devices"

mpirun -n $RANK_SIZE --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --config_path=$CONFIG_FILE \
    --distributed=True \
    --device_target="GPU" \
    --light $LIGHT \
    --lr $LR \
    --loss_scale $LOSS_SCALE \
    --use_global_norm $USE_GLOBAL_NORM \
    --data_path $DATA_PATH &> log &

cd ..
