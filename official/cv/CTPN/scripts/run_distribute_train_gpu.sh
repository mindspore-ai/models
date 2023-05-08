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
# ============================================================================

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash scripts/run_distribute_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH] [CONFIG_PATH](optional)"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

TASK_TYPE=$1
PATH2=$(get_real_path $2)
echo $PATH2
if [ ! -f $PATH2 ]
then 
    echo "error: PRETRAINED_PATH=$PATH2 is not a file"
exit 1
fi
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_PATH=$BASE_PATH/../default_config.yaml
if [ $# == 3 ]
then
    CONFIG_PATH=$(get_real_path $3)
    echo $CONFIG_PATH
fi
rm -rf ./train_parallel
mkdir ./train_parallel
cp ./*.py ./train_parallel
cp ./*yaml ./train_parallel
cp -r ./scripts ./train_parallel
cp -r ./src ./train_parallel
cd ./train_parallel || exit

export DEVICE_NUM=8
export RANK_SIZE=8

echo "start training"
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python train.py --run_distribute=True --task_type=$TASK_TYPE --pre_trained=$PATH2 --device_target="GPU" \
      --config_path=$CONFIG_PATH &> log &
cd ..
