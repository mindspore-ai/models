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

if [ $# != 4 ] && [ $# != 5 ]
then 
    echo "Usage: bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](option)"
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
PATH2=$(get_real_path $2)
PATH3=$(get_real_path $3)
echo $PATH3
echo $PATH1
echo $PATH2

if [ ! -f $PATH1 ]
then
    echo "error: ANNO_PATH=$PATH1 is not a file"
exit 1
fi

if [ ! -f $PATH2 ]
then 
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi 

if [ ! -d $PATH3 ]
then
    echo "error: COCO_ROOT=$PATH3 is not a dir"
exit 1
fi

mindrecord_dir=$PATH3/FASTERRCNN_MINDRECORD/
if [ $# == 5 ]
then
    mindrecord_dir=$(get_real_path $5)
    if [ ! -d $mindrecord_dir ]
    then
        echo "error: mindrecord_dir=$mindrecord_dir is not a dir"
    exit 1
    fi
fi
echo $mindrecord_dir

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config_gpu.yaml"

export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export DEVICE_ID=$4
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log
echo "start eval for device $DEVICE_ID"
python eval.py --config_path=$CONFIG_FILE --device_id=$DEVICE_ID --anno_path=$PATH1 --checkpoint_path=$PATH2 \
--coco_root=$PATH3 --mindrecord_dir=$mindrecord_dir &> log &
cd ..
