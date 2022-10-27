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

if [ $# -le 2 ]
then 
    echo "Usage:
    bash run_standalone_train_gpu.sh [PRETRAINED_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)
    "
exit 1
fi

if [ $2 != "PVAnet" ]
then
  echo "error: the selected backbone must be PVAnet"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $3)
echo $PATH1
echo $PATH2


if [ ! -d $PATH2 ]
then
    echo "error: COCO_ROOT=$PATH2 is not a dir"
exit 1
fi


BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
if [ $# -ge 1 ]; then
  if [ $2 == 'PVAnet' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH
if [ -d "train_${DEVICE_ID}" ];
then
    rm -rf ./train_${DEVICE_ID}
fi
mkdir ./train_${DEVICE_ID}
cp ../train.py ./train_${DEVICE_ID}
cp ../*.yaml ./train_${DEVICE_ID}
cp run_standalone_train_gpu.sh ./train_${DEVICE_ID}
cp -r ../src ./train_${DEVICE_ID}
cd ./train_${DEVICE_ID} || exit
echo "start training for device $DEVICE_ID"
env > env.log
python train.py --config_path=$CONFIG_FILE --coco_root=$PATH2  \
--device_id=$DEVICE_ID --pre_trained=$PATH1 --device_target="GPU" --backbone=$2 &> train.log &

cd ..
