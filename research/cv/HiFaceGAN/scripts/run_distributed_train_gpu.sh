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

if [[ $# != 3 ]]; then
    echo "Usage: bash scripts/run_distributed_train_gpu.sh [RANK_SIZE] [DATA_PATH] [VGG_PATH]"
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
VGG_PATH=$(get_real_path $3)

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a dir"
exit 1
fi

if [ ! -f $VGG_PATH ]
then
    echo "error: VGG_PATH=$VGG_PATH is not a file"
exit 1
fi

if [ -d "scripts/run_distribute_train" ];
then
    rm -rf scripts/run_distribute_train
fi

mkdir scripts/run_distribute_train
cp default_config.yaml train.py scripts/run_distributed_train_gpu.sh scripts/run_distribute_train
cp -r src  scripts/run_distribute_train
cd scripts/run_distribute_train || exit

echo "start training on $RANK_SIZE devices"

mpirun -n $RANK_SIZE --allow-run-as-root \
  --output-filename log_output \
  --merge-stderr-to-stdout \
  python train.py \
  --is_distributed True \
  --device_target GPU \
  --data_path $DATA_PATH \
  --pretrained_vgg_path $VGG_PATH > log 2>&1 &

cd ..
