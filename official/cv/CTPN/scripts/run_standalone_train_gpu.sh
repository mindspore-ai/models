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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_standalone_train.sh [TASK_TYPE] [PRETRAINED_PATH] [DEVICE_ID] [CONFIG_PATH](optional)"
echo "for example: bash run_standalone_train.sh Pretraining /path/vgg16_backbone.ckpt 0"
echo "when device id is occupied, choose for another one"
echo "It is better to use absolute path."
echo "=============================================================================================================="
if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH] [DEVICE_ID] [CONFIG_PATH](optional)"
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
PRETRAINED_PATH=$(get_real_path $2)
echo $PRETRAINED_PATH
if [ ! -f $PRETRAINED_PATH ]
then 
    echo "error: PRETRAINED_PATH=$PRETRAINED_PATH is not a file"
exit 1
fi
BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_PATH=$BASE_PATH/../default_config.yaml
if [ $# == 4 ]
then
    CONFIG_PATH=$(get_real_path $4)
    echo $CONFIG_PATH
fi
rm -rf ./train
mkdir ./train
cp ./*.py ./train
cp ./*yaml ./train
cp -r ./scripts ./train
cp -r ./src ./train
cd ./train || exit

export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

echo "start training for device $3"
export CUDA_VISIBLE_DEVICES=$3
env > env.log
python train.py --task_type=$TASK_TYPE --pre_trained=$PRETRAINED_PATH --device_target="GPU" \
    --config_path=$CONFIG_PATH &> log &
cd ..
