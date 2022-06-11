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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh DEVICE_NUM PRETRAINED_PATH COCO_ROOT MINDRECORD_DIR(optional)"
echo "for example: bash run_distribute_train_gpu.sh 8 /path/pretrain.ckpt cocodataset mindrecord_dir(optional)"
echo "It is better to use absolute path."
echo "=============================================================================================================="

if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)"
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
PRETRAINED_PATH=$(get_real_path $2)
PATH3=$(get_real_path $3)

rm -rf run_distribute_train
mkdir run_distribute_train
cp -rf ../src/ ../train.py ../*.yaml ./run_distribute_train
cd run_distribute_train || exit

mindrecord_dir=$PATH3/FASTERRCNN_MINDRECORD/
if [ $# -eq 4 ]
then
    mindrecord_dir=$(get_real_path $4)
    if [ ! -d $mindrecord_dir ]
    then
        echo "error: mindrecord_dir=$mindrecord_dir is not a dir"
    exit 1
    fi
fi
echo $mindrecord_dir

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../../default_config_gpu.yaml"

echo "start training on $RANK_SIZE devices"

mpirun -n $RANK_SIZE --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python train.py  \
    --config_path=$CONFIG_FILE \
    --run_distribute=True \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --pre_trained=$PRETRAINED_PATH \
    --coco_root=$PATH3 \
    --mindrecord_dir=$mindrecord_dir > log 2>&1 &
