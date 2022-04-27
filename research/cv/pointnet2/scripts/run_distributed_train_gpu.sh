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
# ==========================================================================

PRETRAINED_CKPT=""

if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]"
    echo "============================================================"
    echo "[DEVICE_NUM]: The number of cuda devices to use."
    echo "[DATA_PATH]: The path to the train and evaluation datasets."
    echo "[SAVE_DIR]: The path to save files generated during training."
    echo "[PRETRAINDE_CKPT]: (optional) The path to the checkpoint file."
    echo "============================================================"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# -ge 3 ]
then
    DATA_PATH=$(get_real_path $2)
    SAVE_DIR=$(get_real_path $3)

    if [ ! -d $DATA_PATH ]
    then
        echo "error: DATA_PATH=$DATA_PATH is not a directory"
    exit 1
    fi
fi

if [ $# -ge 4 ]
then
    PRETRAINED_CKPT=$(get_real_path $4)
    if [ ! -f $PRETRAINED_CKPT ]
    then
        echo "error: PRETRAINED_CKPT=$PRETRAINED_CKPT is not a file"
    exit 1
    fi
fi

ulimit -u unlimited
export DEVICE_NUM=$1
export RANK_SIZE=$1

mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python train.py \
      --platform=GPU \
      --data_path=$DATA_PATH \
      --pretrained_ckpt=$PRETRAINED_CKPT \
      --save_dir=$SAVE_DIR > train_dis.log 2>&1 &
